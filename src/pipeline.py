"""Inference-time pipeline on top of a trained D-FINE: flip TTA, tiled inference, box fusion, temporal rescoring.

Detections are kept as a column table ("Dets": a dict of equally long numpy arrays) rather than one dict per box,
because tiled + flipped inference yields ~3,000 boxes per frame (~10M on the validation sequences):
    frame  int32    index into the `frames` list the predictions were made for
    view   int16    0 = full frame, 1.. = tile (row-major)
    flip   bool     predicted on the horizontally flipped view (boxes are already flipped back)
    x1 y1 x2 y2     float32, original image pixels
    score  float32
    label  int16    COCO category id 1..8

`frames` are dicts with "id" (COCO image id or test image_id), "file_name" (relative to the dataset root),
"width", "height", "seq" and "frame_no"; see val_frames() / test_frames().
"""

import contextlib
import io
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.dfine import ImgSize, _hw
from src.evaluation import evaluate_coco_metrics
from src.preprocessing import letterbox
from src.submission import build_submission, load_test_frames

Dets = Dict[str, np.ndarray]
_DTYPES = {
    "frame": np.int32, "view": np.int16, "flip": bool,
    "x1": np.float32, "y1": np.float32, "x2": np.float32, "y2": np.float32,
    "score": np.float32, "label": np.int16,
}


# --------------------------------------------------------------------------- frames and detection tables


def val_frames(dataset_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """All validation frames (every frame, sorted by sequence and frame number)."""
    coco = COCO(str(Path(dataset_dir) / "val" / "annotations_coco.json"))
    imgs = sorted(coco.dataset["images"], key=lambda im: (im["video"], int(im["frame_id"])))
    return [
        {"id": im["id"], "file_name": f"val/{im['file_name']}", "width": im["width"], "height": im["height"],
         "seq": im["video"], "frame_no": int(im["frame_id"])}
        for im in imgs
    ]


def test_frames(dataset_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """All test frames in test_images.csv order (the order build_submission writes)."""
    return [{**f, "seq": f["clip"], "frame_no": int(f["frame"])} for f in load_test_frames(dataset_dir)]


def empty_dets() -> Dets:
    return {k: np.zeros(0, dtype=t) for k, t in _DTYPES.items()}


def concat_dets(parts: Sequence[Dets]) -> Dets:
    return {k: np.concatenate([p[k] for p in parts]).astype(t, copy=False) for k, t in _DTYPES.items()}


def select(d: Dets, mask: np.ndarray) -> Dets:
    return {k: v[mask] for k, v in d.items()}


def save_dets(path: Union[str, Path], d: Dets) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


def load_dets(path: Union[str, Path]) -> Dets:
    with np.load(path) as z:
        return {k: z[k].astype(t, copy=False) for k, t in _DTYPES.items()}


def cached(path: Union[str, Path], compute: Callable[[], Dets]) -> Dets:
    """Loads `path` if it exists, otherwise computes, saves and returns the detections."""
    path = Path(path)
    if path.exists():
        print(f"loaded {path.name}")
        return load_dets(path)
    d = compute()
    save_dets(path, d)
    return d


def _boxes(d: Dets) -> np.ndarray:
    return np.stack([d["x1"], d["y1"], d["x2"], d["y2"]], 1)


def top_per_frame(d: Dets, k: int = 500) -> Dets:
    """Keeps the k highest-scoring detections of every frame (the scorer ignores the rest)."""
    if len(d["score"]) == 0:
        return d
    order = np.lexsort((-d["score"], d["frame"]))
    frames = d["frame"][order]
    first = np.r_[0, np.flatnonzero(np.diff(frames)) + 1]
    start_of_group = np.repeat(first, np.diff(np.r_[first, len(frames)]))
    rank = np.arange(len(frames)) - start_of_group
    return select(d, order[rank < k])


# --------------------------------------------------------------------------- views (full frame, tiles, flips)


def tile_grid(width: int, height: int, grid: Tuple[int, int], overlap: float) -> List[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of a rows x cols grid of equally sized tiles that overlap by `overlap` of a tile side.
    Tiles keep the frame's aspect ratio, so they letterbox into the model input without extra padding."""
    rows, cols = grid
    tw = int(round(width / (cols - (cols - 1) * overlap))) if cols > 1 else width
    th = int(round(height / (rows - (rows - 1) * overlap))) if rows > 1 else height
    # last row / column pinned to the frame edge so rounding never leaves an uncovered strip
    xs = [width - tw if c == cols - 1 else min(int(round(c * tw * (1 - overlap))), width - tw) for c in range(cols)]
    ys = [height - th if r == rows - 1 else min(int(round(r * th * (1 - overlap))), height - th) for r in range(rows)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


class ViewsDataset(Dataset):
    """Per frame: the letterboxed views to predict (full frame and/or tiles, optionally flipped).
    The image is decoded once per frame; items are (list of tensors, list of view metas)."""

    def __init__(
        self,
        frames: List[Dict[str, Any]],
        image_root: Union[str, Path],
        imgsz: ImgSize,
        full: bool = True,
        grid: Optional[Tuple[int, int]] = None,
        overlap: float = 0.2,
        flip: bool = False,
    ) -> None:
        self.frames, self.image_root, self.hw = frames, Path(image_root), _hw(imgsz)
        self.full, self.grid, self.overlap, self.flip = full, grid, overlap, flip

    def __len__(self) -> int:
        return len(self.frames)

    def views(self, width: int, height: int) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """(view id, crop) pairs: 0 = full frame, 1.. = tiles."""
        out = [(0, (0, 0, width, height))] if self.full else []
        if self.grid is not None:
            out += [(i + 1, c) for i, c in enumerate(tile_grid(width, height, self.grid, self.overlap))]
        return out

    def __getitem__(self, i: int):
        img = cv2.cvtColor(cv2.imread(str(self.image_root / self.frames[i]["file_name"])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        tensors, metas = [], []
        for view, (x0, y0, x1, y1) in self.views(W, H):
            crop = img[y0:y1, x0:x1]
            if self.flip:
                crop = crop[:, ::-1]
            padded, r, (dw, dh) = letterbox(np.ascontiguousarray(crop), target_shape=self.hw)
            tensors.append(torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1) / 255.0))
            metas.append({"frame": i, "view": view, "flip": self.flip, "x0": x0, "y0": y0, "cw": x1 - x0,
                          "ch": y1 - y0, "W": W, "H": H, "r": r, "dw": dw, "dh": dh})
        return tensors, metas


def _collate_views(batch):
    return [t for ts, _ in batch for t in ts], [m for _, ms in batch for m in ms]


def letterbox_to_frame(xyxy: np.ndarray, meta: Dict[str, Any], border_margin: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Boxes predicted on one letterboxed view (pixels of the model input) -> original frame pixels.

    Undoes the letterbox, the flip and the crop offset, clips to the crop. With `border_margin` set, boxes of a
    tile touching an edge of the tile that lies inside the frame are flagged (`keep` False): they are usually
    objects cut by the tile, which the neighbouring tile or the full-frame view sees whole.
    Returns (boxes (N, 4) in frame pixels, keep mask (N,)).
    """
    b = xyxy.astype(np.float32).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - meta["dw"]) / meta["r"]
    b[:, [1, 3]] = (b[:, [1, 3]] - meta["dh"]) / meta["r"]
    cw, ch = meta["cw"], meta["ch"]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, cw)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, ch)
    if meta["flip"]:
        b[:, [0, 2]] = cw - b[:, [2, 0]]
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    if border_margin is not None and meta["view"] > 0:
        m = border_margin
        x0, y0 = meta["x0"], meta["y0"]
        if x0 > 0:
            keep &= b[:, 0] > m
        if y0 > 0:
            keep &= b[:, 1] > m
        if x0 + cw < meta["W"]:
            keep &= b[:, 2] < cw - m
        if y0 + ch < meta["H"]:
            keep &= b[:, 3] < ch - m
    b[:, [0, 2]] += meta["x0"]
    b[:, [1, 3]] += meta["y0"]
    return b, keep


@torch.no_grad()
def predict_views(
    model: torch.nn.Module,
    frames: List[Dict[str, Any]],
    image_root: Union[str, Path],
    imgsz: ImgSize,
    device: Any,
    full: bool = True,
    grid: Optional[Tuple[int, int]] = None,
    overlap: float = 0.2,
    flip: bool = False,
    max_det: int = 500,
    score_thresh: float = 0.001,
    border_margin: Optional[float] = 2.0,
    batch: int = 4,
    workers: int = 4,
    amp: bool = True,
) -> Dets:
    """Predicts every view of every frame; `max_det` top (query, class) pairs per view, boxes in frame pixels."""
    model.eval()
    dev = torch.device(device)
    use_amp = amp and dev.type == "cuda"
    H, W = _hw(imgsz)
    ds = ViewsDataset(frames, image_root, imgsz, full=full, grid=grid, overlap=overlap, flip=flip)
    views_per_frame = (1 if full else 0) + (grid[0] * grid[1] if grid else 0)
    loader = DataLoader(
        ds, batch_size=max(1, batch // views_per_frame), shuffle=False, num_workers=workers,
        collate_fn=_collate_views, pin_memory=dev.type == "cuda", persistent_workers=False,
    )
    parts: List[Dets] = []
    t0, n_views = time.perf_counter(), 0
    for tensors, metas in tqdm(loader, desc=f"predict ({'full' if full else ''}{'+tiles' if grid else ''}{' flip' if flip else ''})"):
        for s in range(0, len(tensors), batch):
            chunk, chunk_meta = tensors[s:s + batch], metas[s:s + batch]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(pixel_values=torch.stack(chunk).to(dev, non_blocking=True))
            prob = out.logits.sigmoid().float()
            n_cls = prob.shape[-1]
            scores, idx = prob.flatten(1).topk(min(max_det, prob.shape[1] * n_cls), dim=1)
            labels, queries = idx % n_cls, idx // n_cls
            bx = out.pred_boxes.float().gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
            cx, cy, bw, bh = bx.unbind(-1)
            xyxy = torch.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], -1)
            xyxy, scores, labels = xyxy.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
            for j, meta in enumerate(chunk_meta):
                b, keep = letterbox_to_frame(xyxy[j], meta, border_margin)
                keep &= scores[j] >= score_thresh
                n = int(keep.sum())
                parts.append({
                    "frame": np.full(n, meta["frame"]), "view": np.full(n, meta["view"]), "flip": np.full(n, flip),
                    "x1": b[keep, 0], "y1": b[keep, 1], "x2": b[keep, 2], "y2": b[keep, 3],
                    "score": scores[j][keep], "label": labels[j][keep] + 1,
                })
            n_views += len(chunk)
    elapsed = time.perf_counter() - t0
    print(f"{len(frames):,} frames, {n_views:,} views in {elapsed / 60:.1f} min ({1000 * elapsed / max(1, len(frames)):.0f} ms/frame)")
    return concat_dets(parts) if parts else empty_dets()


PASS_KEYS = ("full", "full_flip", "tiles", "tiles_flip")


def predict_passes(
    model: torch.nn.Module,
    frames: List[Dict[str, Any]],
    image_root: Union[str, Path],
    imgsz: ImgSize,
    device: Any,
    cache_dir: Union[str, Path],
    split: str,
    keys: Sequence[str] = PASS_KEYS,
    grid: Tuple[int, int] = (2, 2),
    overlap: float = 0.2,
    full_max_det: int = 500,
    tile_max_det: int = 300,
    border_margin: Optional[float] = 2.0,
    batch: int = 4,
    workers: int = 4,
    amp: bool = True,
) -> Dict[str, Dets]:
    """Runs (or loads from cache_dir) the prediction passes `keys` out of PASS_KEYS for `frames`.
    Cache file names encode the pass settings, so changing them triggers a new prediction."""
    out: Dict[str, Dets] = {}
    for key in keys:
        tiles, flip = key.startswith("tiles"), key.endswith("_flip")
        tag = (f"tiles{grid[0]}x{grid[1]}_ov{overlap}_md{tile_max_det}_bm{border_margin}" if tiles
               else f"full_md{full_max_det}") + ("_flip" if flip else "")
        out[key] = cached(
            Path(cache_dir) / f"{split}_{tag}.npz",
            lambda: predict_views(
                model, frames, image_root, imgsz, device, full=not tiles, grid=grid if tiles else None,
                overlap=overlap, flip=flip, max_det=tile_max_det if tiles else full_max_det,
                border_margin=border_margin, batch=batch, workers=workers, amp=amp,
            ),
        )
        print(f"{split} {key}: {len(out[key]['score']):,} boxes")
    return out


# --------------------------------------------------------------------------- box fusion


def _iou_one(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(b[2], bs[:, 2]) - np.maximum(b[0], bs[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], bs[:, 3]) - np.maximum(b[1], bs[:, 1]), 0, None)
    inter = ix * iy
    union = (b[2] - b[0]) * (b[3] - b[1]) + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def _wbf_group(b: np.ndarray, s: np.ndarray, src: np.ndarray, iou_thr: float, conf_type: str, n_sources: int):
    """Weighted boxes fusion of one group (same frame and class). A cluster takes at most one box per source
    (`src` = 0..n_sources-1): one prediction pass contains each object once, so two overlapping boxes of the
    same pass are different objects (e.g. people in a crowd), not duplicates.
    Returns (cluster boxes, scores, member -> cluster)."""
    order = np.argsort(-s)
    fused = np.empty((len(b), 4), np.float64)
    wsum = np.empty((len(b), 4), np.float64)
    ssum, smax, cnt = np.empty(len(b)), np.empty(len(b)), np.empty(len(b))
    has_src = np.zeros((len(b), max(n_sources, int(src.max()) + 1)), bool)
    owner = np.empty(len(b), np.int64)
    m = 0
    for i in order:
        if m:
            ious = _iou_one(b[i], fused[:m])
            ious[has_src[:m, src[i]]] = -1.0
            j = int(ious.argmax())
            if ious[j] > iou_thr:
                wsum[j] += b[i] * s[i]
                ssum[j] += s[i]
                smax[j] = max(smax[j], s[i])
                cnt[j] += 1
                fused[j] = wsum[j] / ssum[j]
                has_src[j, src[i]] = True
                owner[i] = j
                continue
        fused[m], wsum[m], ssum[m], smax[m], cnt[m], owner[i] = b[i], b[i] * s[i], s[i], s[i], 1, m
        has_src[m, src[i]] = True
        m += 1
    if conf_type == "avg":  # as ensemble_boxes: boxes found by fewer than n_sources sources are down-weighted
        score = ssum[:m] / cnt[:m] * np.minimum(cnt[:m], n_sources) / n_sources
    else:
        score = smax[:m]
    return fused[:m], score, owner


def fuse(
    d: Dets,
    by: Sequence[str] = ("frame", "label"),
    method: str = "wbf",
    iou_thr: float = 0.55,
    conf_type: str = "avg",
    n_sources: int = 1,
    source: Optional[str] = None,
) -> Dets:
    """Merges duplicate detections within groups of equal `by` columns (always at least frame and label).

    method="nms":  per-group NMS, keeps the highest-scoring box of each cluster unchanged.
    method="wbf":  weighted boxes fusion; cluster box = score-weighted mean of its members,
                   score = mean member score scaled by min(members, n_sources) / n_sources (conf_type="avg")
                   or the max member score (conf_type="max"). With `source` (a column such as "flip" or "view"),
                   a cluster holds at most one box per source value, so only boxes of different passes/views merge.
    The view/flip columns of a fused box are those of its highest-scoring member.
    """
    by = tuple(dict.fromkeys(("frame", "label") + tuple(by)))
    if len(d["score"]) == 0:
        return d
    keys = np.stack([d[k].astype(np.int64) for k in by], 1)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.ravel()
    boxes = _boxes(d).astype(np.float64)

    if method == "nms":
        keep = torchvision.ops.batched_nms(
            torch.from_numpy(boxes), torch.from_numpy(d["score"].astype(np.float64)), torch.from_numpy(group), iou_thr
        ).numpy()
        return select(d, np.sort(keep))
    if method != "wbf":
        raise ValueError(f"unknown method {method!r}")

    src_ids = (np.unique(d[source], return_inverse=True)[1].ravel() if source is not None
               else np.zeros(len(d["score"]), np.int64))
    order = np.argsort(group, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(group[order])) + 1, len(order)]
    out_boxes, out_scores, out_src = [], [], []
    for a, z in zip(bounds[:-1], bounds[1:]):
        idx = order[a:z]
        if len(idx) == 1:
            out_boxes.append(boxes[idx])
            out_scores.append(d["score"][idx].astype(np.float64) * (1.0 / n_sources if conf_type == "avg" else 1.0))
            out_src.append(idx)
            continue
        fb, fs, owner = _wbf_group(boxes[idx], d["score"][idx].astype(np.float64), src_ids[idx], iou_thr, conf_type, n_sources)
        # representative member (highest score) of every cluster, for the frame/label/view/flip columns
        rep = np.full(len(fb), -1)
        for member in np.argsort(-d["score"][idx]):
            if rep[owner[member]] < 0:
                rep[owner[member]] = member
        out_boxes.append(fb)
        out_scores.append(fs)
        out_src.append(idx[rep])
    src = np.concatenate(out_src)
    fb = np.concatenate(out_boxes)
    res = {k: d[k][src] for k in ("frame", "view", "flip", "label")}
    res.update({"x1": fb[:, 0], "y1": fb[:, 1], "x2": fb[:, 2], "y2": fb[:, 3], "score": np.concatenate(out_scores)})
    return {k: res[k].astype(t, copy=False) for k, t in _DTYPES.items()}


def combine(
    passes: Dict[str, Dets],
    flip: bool = False,
    tiles: bool = False,
    flip_iou: float = 0.55,
    merge: str = "wbf",
    merge_iou: float = 0.55,
    max_det: int = 500,
) -> Dets:
    """Builds one prediction set from cached passes ("full", "full_flip", "tiles", "tiles_flip").

    1. flip=True: every view is fused with its flipped twin (WBF, at most one box of each, scores averaged over
       the 2 predictions, so a box found in only one of them keeps half its score).
    2. tiles=True: full-frame and tile detections are merged per frame and class with `merge`:
       "wbf" fuses boxes of different views only (at most one per view) and keeps the max score; scores are not
       averaged because small objects are often found by one tile only, and averaging would penalise exactly them.
       "nms" is plain per-class NMS over all views (also suppresses overlapping objects within one view).
    3. The top `max_det` boxes per frame are kept.
    """
    parts = ["full"] + (["tiles"] if tiles else [])
    sets = []
    for p in parts:
        if flip:
            sets.append(fuse(concat_dets([passes[p], passes[p + "_flip"]]), by=("frame", "view", "label"),
                             method="wbf", iou_thr=flip_iou, conf_type="avg", n_sources=2, source="flip"))
        else:
            sets.append(passes[p])
    d = concat_dets(sets)
    if tiles:
        d = fuse(d, by=("frame", "label"), method=merge, iou_thr=merge_iou, conf_type="max", source="view")
    return top_per_frame(d, max_det)


# --------------------------------------------------------------------------- temporal rescoring


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def temporal_rescore(
    d: Dets,
    frames: List[Dict[str, Any]],
    window: int = 2,
    iou_thr: float = 0.5,
    weight: float = 0.5,
) -> Dets:
    """Re-scores every detection with the support it gets from the neighbouring frames of its sequence.

    For each neighbour frame t+k (0 < |k| <= window, same sequence), the support is the highest score among
    same-class detections overlapping the box with IoU >= iou_thr (0 if none). New score =
    (1 - weight) * score + weight * mean support over the available neighbours. Objects persist between
    consecutive frames while many false positives flicker, so this mostly re-ranks detections; boxes are unchanged.
    Frames at sequence ends average over the neighbours they have.
    """
    if len(d["score"]) == 0 or window <= 0 or weight <= 0:
        return d
    lookup = {(f["seq"], f["frame_no"]): i for i, f in enumerate(frames)}
    order = np.argsort(d["frame"], kind="stable")
    d = select(d, order)
    starts = np.searchsorted(d["frame"], np.arange(len(frames) + 1))
    boxes = _boxes(d).astype(np.float64)
    new_scores = d["score"].astype(np.float64).copy()
    offsets = [k for k in range(-window, window + 1) if k != 0]
    for fi, f in enumerate(tqdm(frames, desc="temporal", leave=False)):
        a, z = starts[fi], starts[fi + 1]
        if a == z:
            continue
        bt, lt, st = boxes[a:z], d["label"][a:z], d["score"][a:z].astype(np.float64)
        support_sum, n_nb = np.zeros(z - a), 0
        for k in offsets:
            ni = lookup.get((f["seq"], f["frame_no"] + k))
            if ni is None:
                continue
            n_nb += 1
            na, nz = starts[ni], starts[ni + 1]
            if na == nz:
                continue
            match = (_iou_matrix(bt, boxes[na:nz]) >= iou_thr) & (lt[:, None] == d["label"][None, na:nz])
            support_sum += np.where(match, d["score"][None, na:nz], 0.0).max(1)
        if n_nb:
            new_scores[a:z] = (1 - weight) * st + weight * support_sum / n_nb
    d["score"] = new_scores.astype(np.float32)
    return d


# --------------------------------------------------------------------------- evaluation and submission


def to_records(d: Dets, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """COCO result records (bbox [x, y, w, h]) with image_id = frames[i]["id"]."""
    ids = [f["id"] for f in frames]
    return [
        {"image_id": ids[fi], "category_id": int(c), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
         "score": float(s)}
        for fi, c, x1, y1, x2, y2, s in zip(d["frame"], d["label"], d["x1"], d["y1"], d["x2"], d["y2"], d["score"])
    ]


def evaluate_dets(
    d: Dets,
    frames: List[Dict[str, Any]],
    coco_gt: COCO,
    eval_ids: Sequence[int],
    max_det: int = 500,
    quiet: bool = True,
) -> Dict[str, float]:
    """COCO metrics on the frames whose id is in eval_ids (top max_det boxes per frame)."""
    eval_set = set(eval_ids)
    in_eval = np.array([f["id"] in eval_set for f in frames])
    records = to_records(top_per_frame(select(d, in_eval[d["frame"]]), max_det), frames)
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        return evaluate_coco_metrics(coco_gt, records, image_ids=list(eval_ids), max_dets=max_det)


def write_submission(d: Dets, frames: List[Dict[str, Any]], path: Union[str, Path], max_det: int = 500):
    """Submission CSV (test_images.csv order, top max_det boxes per frame)."""
    return build_submission(to_records(top_per_frame(d, max_det), frames), frames, path, max_dets=max_det)

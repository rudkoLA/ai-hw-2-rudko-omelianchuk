"""Optical flow temporal motion prediction, bounding box refinement, and gap filling.

Integrates dense Farneback optical flow with the detection table format (Dets) from src.pipeline:
1. compute_optical_flow: calculates dense optical flow between two consecutive frames on
   downscaled grayscale images (scale=0.25 by default) for high throughput.
2. warp_boxes: propagates bounding boxes across frames using flow median velocity sampled
   inside the object's inner core to avoid background motion clutter.
3. optical_flow_refine:
   - Flow-guided temporal coordinate smoothing (blending single-frame detections with
     motion-predicted bounding boxes from adjacent frames).
   - Motion-aligned temporal rescoring (evaluating inter-frame IoU after motion compensation).
   - Optional gap-filling recovery (interpolating confident tracks across dropped frames).
"""

from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.pipeline import Dets, empty_dets, concat_dets, select, top_per_frame, _boxes, _iou_matrix, _DTYPES


def compute_frame_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    scale: float = 0.25,
) -> np.ndarray:
    """Computes dense Farneback optical flow from prev_gray to curr_gray.
    Images are resized by `scale` to accelerate processing.
    Returns flow field (H_s, W_s, 2) in scaled pixels.
    """
    if scale != 1.0:
        p_small = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        c_small = cv2.resize(curr_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        p_small, c_small = prev_gray, curr_gray

    return cv2.calcOpticalFlowFarneback(
        p_small, c_small, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def warp_boxes(
    boxes: np.ndarray,
    flow_small: np.ndarray,
    scale: float,
    width: int,
    height: int,
    inner_margin: float = 0.15,
) -> np.ndarray:
    """Propagates bounding boxes [x1, y1, x2, y2] along dense optical flow field `flow_small`.

    Args:
        boxes: (N, 4) in full-resolution image pixels.
        flow_small: (H_s, W_s, 2) dense flow in scaled pixel coordinates.
        scale: downscaling factor used when computing flow_small.
        width: full-resolution frame width.
        height: full-resolution frame height.
        inner_margin: fractional margin to sample flow from the object core.

    Returns:
        (N, 4) warped bounding boxes clipped to [0, width] and [0, height].
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    h_s, w_s = flow_small.shape[:2]
    warped = boxes.copy().astype(np.float32)

    x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ws = np.maximum(x2s - x1s, 1.0)
    hs = np.maximum(y2s - y1s, 1.0)

    # Core patch coordinates on the scaled flow map
    cx1s = np.clip(np.round((x1s + inner_margin * ws) * scale).astype(int), 0, w_s - 1)
    cx2s = np.clip(np.round((x2s - inner_margin * ws) * scale).astype(int), cx1s + 1, w_s)
    cy1s = np.clip(np.round((y1s + inner_margin * hs) * scale).astype(int), 0, h_s - 1)
    cy2s = np.clip(np.round((y2s - inner_margin * hs) * scale).astype(int), cy1s + 1, h_s)

    for i in range(len(boxes)):
        patch = flow_small[cy1s[i]:cy2s[i], cx1s[i]:cx2s[i]]
        if patch.size > 0:
            dx = float(np.median(patch[:, :, 0])) / scale
            dy = float(np.median(patch[:, :, 1])) / scale
            warped[i, [0, 2]] += dx
            warped[i, [1, 3]] += dy

    warped[:, [0, 2]] = np.clip(warped[:, [0, 2]], 0, width)
    warped[:, [1, 3]] = np.clip(warped[:, [1, 3]], 0, height)
    return warped


def load_grayscale_frame(image_path: Union[str, Path]) -> np.ndarray:
    """Reads an image as single-channel grayscale."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return img


def optical_flow_refine(
    d: Dets,
    frames: List[Dict[str, Any]],
    dataset_dir: Union[str, Path],
    smooth_weight: float = 0.25,
    rescore_weight: float = 0.25,
    iou_thr: float = 0.50,
    window: int = 1,
    gap_fill: bool = False,
    flow_scale: float = 0.25,
    cache_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dets:
    """Refines detections across video sequences using optical-flow temporal motion compensation.

    1. Motion Compensation: For each adjacent frame t+k (|k| <= window), computes optical flow
       to warp bounding boxes to frame t.
    2. Coordinate Smoothing: Blends the detector's box at frame t with flow-warped predictions
       from adjacent frames: B_t = (1 - smooth_weight) * B_t + smooth_weight * mean(B_warped).
    3. Motion-Aligned Rescoring: Re-scores detections based on temporal support of flow-aligned
       matches rather than static box overlap.
    4. Gap Filling (optional): Recovers high-confidence objects confirmed at t-1 and t+1 that
       were momentarily dropped at frame t.

    Args:
        d: Dets table dictionary.
        frames: List of frame metadata dicts (file_name, seq, frame_no, width, height).
        dataset_dir: Root dataset directory.
        smooth_weight: Coordinate smoothing weight (0 = no coordinate change).
        rescore_weight: Temporal score rescoring weight (0 = no score change).
        iou_thr: IoU threshold for matching current detection with warped neighbor detection.
        window: Temporal radius in frames (1 = immediate neighbours t-1, t+1).
        gap_fill: If True, interpolates confident missing detections across 1-frame gaps.
        flow_scale: Downscale factor for Farneback optical flow (0.25 = 4x downscale).
        cache_dir: Optional folder to cache calculated optical flow fields.
        verbose: Displays progress bar.

    Returns:
        Refined Dets table dictionary.
    """
    if len(d["score"]) == 0 or (smooth_weight <= 0 and rescore_weight <= 0 and not gap_fill):
        return d

    dataset_dir = Path(dataset_dir)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    lookup = {(f["seq"], f["frame_no"]): i for i, f in enumerate(frames)}
    order = np.argsort(d["frame"], kind="stable")
    d = select(d, order)
    starts = np.searchsorted(d["frame"], np.arange(len(frames) + 1))
    boxes = _boxes(d).astype(np.float64)
    scores = d["score"].astype(np.float64)
    labels = d["label"]

    new_boxes = boxes.copy()
    new_scores = scores.copy()

    # Pre-load or cache optical flows per consecutive pair within each sequence
    flow_cache: Dict[Tuple[str, int, int], np.ndarray] = {}
    gray_cache: Dict[str, np.ndarray] = {}

    def get_scaled_gray(rel_path: str) -> np.ndarray:
        if rel_path in gray_cache:
            return gray_cache[rel_path]
        full_p = dataset_dir / rel_path
        img = cv2.imread(str(full_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {full_p}")
        if flow_scale != 1.0:
            small = cv2.resize(img, (0, 0), fx=flow_scale, fy=flow_scale, interpolation=cv2.INTER_AREA)
        else:
            small = img
        gray_cache[rel_path] = small
        if len(gray_cache) > 200:
            # Evict oldest frame to keep memory light
            del gray_cache[next(iter(gray_cache))]
        return small

    def get_flow(seq: str, f_from: int, f_to: int, path_from: str, path_to: str) -> np.ndarray:
        key = (seq, f_from, f_to)
        if key in flow_cache:
            return flow_cache[key]
        if cache_dir is not None:
            cfile = cache_dir / f"flow_{seq}_{f_from}_{f_to}_s{flow_scale}.npy"
            if cfile.exists():
                arr = np.load(cfile)
                flow_cache[key] = arr
                return arr

        im_from = get_scaled_gray(path_from)
        im_to = get_scaled_gray(path_to)
        flow = cv2.calcOpticalFlowFarneback(
            im_from, im_to, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_cache[key] = flow
        if cache_dir is not None:
            np.save(cfile, flow)
        return flow

    offsets = [k for k in range(-window, window + 1) if k != 0]

    # Process sequence by sequence, frame by frame
    pbar = tqdm(enumerate(frames), total=len(frames), desc="optical flow refine", leave=False, disable=not verbose)
    for fi, f in pbar:
        a, z = starts[fi], starts[fi + 1]
        bt, lt, st = boxes[a:z], labels[a:z], scores[a:z]
        W, H = f["width"], f["height"]

        if a < z:
            coords_accum = bt.copy()
            coords_weight_sum = np.ones(z - a, dtype=np.float64)
            support_sum = np.zeros(z - a, dtype=np.float64)
            n_support = 0

            for k in offsets:
                ni = lookup.get((f["seq"], f["frame_no"] + k))
                if ni is None:
                    continue
                na, nz = starts[ni], starts[ni + 1]
                if na == nz:
                    continue

                fn = frames[ni]
                # Flow from neighbour frame to current frame
                # Notice: get_flow(from fn to f) produces flow field pointing from fn -> f
                flow = get_flow(f["seq"], fn["frame_no"], f["frame_no"], fn["file_name"], f["file_name"])

                # Warp neighbour boxes to current frame coordinates
                warped_nb_boxes = warp_boxes(boxes[na:nz], flow, scale=flow_scale, width=W, height=H)
                nb_labels = labels[na:nz]
                nb_scores = scores[na:nz]

                # Match warped boxes with current detections
                ious = _iou_matrix(bt, warped_nb_boxes)
                match = (ious >= iou_thr) & (lt[:, None] == nb_labels[None, :])

                if smooth_weight > 0:
                    for i in range(z - a):
                        m_idx = np.flatnonzero(match[i])
                        if len(m_idx) > 0:
                            # Take highest scoring match from this neighbour
                            best_j = m_idx[np.argmax(nb_scores[m_idx])]
                            coords_accum[i] += warped_nb_boxes[best_j]
                            coords_weight_sum[i] += 1.0

                if rescore_weight > 0:
                    n_support += 1
                    support_sum += np.where(match, nb_scores[None, :], 0.0).max(1)

            if smooth_weight > 0:
                smoothed = coords_accum / coords_weight_sum[:, None]
                new_boxes[a:z] = (1.0 - smooth_weight) * bt + smooth_weight * smoothed

            if rescore_weight > 0 and n_support > 0:
                new_scores[a:z] = (1.0 - rescore_weight) * st + rescore_weight * (support_sum / n_support)

    out = {
        "frame": d["frame"],
        "view": d["view"],
        "flip": d["flip"],
        "x1": new_boxes[:, 0].astype(np.float32),
        "y1": new_boxes[:, 1].astype(np.float32),
        "x2": new_boxes[:, 2].astype(np.float32),
        "y2": new_boxes[:, 3].astype(np.float32),
        "score": new_scores.astype(np.float32),
        "label": d["label"],
    }
    return {k: out[k].astype(t, copy=False) for k, t in _DTYPES.items()}

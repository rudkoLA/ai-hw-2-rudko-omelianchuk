"""D-FINE (Hugging Face transformers) training, evaluation and prediction on the shared VisDrone protocol.

Data comes from VisDroneDataset (letterbox + src.preprocessing augmentations, no ImageNet normalisation,
matching D-FINE's own preprocessing), predictions are mapped back to original pixels with
format_coco_predictions and scored with evaluate_coco_metrics.

Run directory layout (mirrors the Ultralytics notebooks):
    last.pt            model / EMA / optimizer / scaler / scheduler state, written every epoch
    best/              best EMA weights in Hugging Face format (DFineForObjectDetection.from_pretrained)
    results.csv        per-epoch train losses, val losses and val COCO metrics
    train_state.json   {"epoch", "best_map", "finished"}
"""

import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import DFineForObjectDetection

from src.dataset import VisDroneDataset, detection_collate_fn
from src.evaluation import evaluate_coco_metrics, format_coco_predictions
from src.preprocessing import get_train_transforms, letterbox

LOSS_TYPES = ("vfl", "bbox", "giou", "fgl", "ddf")


# --------------------------------------------------------------------------- model


def build_dfine(checkpoint: str, class_names: List[str], **config_overrides: Any) -> DFineForObjectDetection:
    """Loads a pretrained D-FINE with a new 8-class head. `config_overrides` are DFineConfig fields,
    e.g. num_queries, num_denoising, matcher_class_cost, weight_loss_vfl."""
    return DFineForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=len(class_names),
        id2label=dict(enumerate(class_names)),
        label2id={n: i for i, n in enumerate(class_names)},
        ignore_mismatched_sizes=True,
        **config_overrides,
    )


def load_dfine(path: Union[str, Path], device: Any) -> DFineForObjectDetection:
    """Loads a saved D-FINE (e.g. run_dir / "best") in eval mode."""
    return DFineForObjectDetection.from_pretrained(str(path)).to(device).eval()


class ModelEMA:
    """Exponential moving average of model weights with warm-up (as in the official D-FINE training)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, warmups: int = 1000) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmups, self.updates = decay, warmups, 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmups))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


# --------------------------------------------------------------------------- data


class FramesDataset(Dataset):
    """Letterboxed images for prediction only. `frames` are dicts with "id" and "file_name"
    (relative to `image_root`): COCO image records or src.submission.load_test_frames()."""

    def __init__(self, frames: List[Dict[str, Any]], image_root: Union[str, Path], imgsz: int) -> None:
        self.frames, self.image_root, self.imgsz = frames, Path(image_root), imgsz

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        frame = self.frames[i]
        img = cv2.cvtColor(cv2.imread(str(self.image_root / frame["file_name"])), cv2.COLOR_BGR2RGB)
        padded, r, (dw, dh) = letterbox(img, target_shape=(self.imgsz, self.imgsz))
        tensor = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1) / 255.0)
        meta = {"image_id": frame["id"], "scale_ratio": r, "pad": (dw, dh), "orig_size": img.shape[:2]}
        return tensor, meta


def _meta(target: Dict[str, Any]) -> Dict[str, Any]:
    """VisDroneDataset target -> the same meta dict FramesDataset returns."""
    if "boxes" not in target:
        return target
    return {
        "image_id": int(target["image_id"][0]),
        "scale_ratio": float(target["scale_ratio"][0]),
        "pad": (float(target["pad"][0]), float(target["pad"][1])),
        "orig_size": (int(target["orig_size"][0]), int(target["orig_size"][1])),
    }


def _to_labels(targets, imgsz: int, device: Any) -> List[Dict[str, torch.Tensor]]:
    """VisDroneDataset targets (letterbox xyxy pixels, COCO ids 1..8) -> D-FINE labels
    (normalised cx cy w h, class ids 0..7)."""
    labels = []
    for t in targets:
        b = t["boxes"] / imgsz
        cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        labels.append({"class_labels": (t["labels"] - 1).to(device), "boxes": cxcywh.to(device)})
    return labels


def _loader(dataset: Dataset, batch: int, workers: int, shuffle: bool, seed: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


# --------------------------------------------------------------------------- prediction


def _postprocess(logits, pred_boxes, metas, imgsz: int, max_det: int, score_thresh: float) -> List[Dict[str, Any]]:
    """Top-k (query, class) pairs per image -> COCO records in original pixel coordinates."""
    prob = logits.sigmoid().float()
    n_cls = prob.shape[-1]
    scores, idx = prob.flatten(1).topk(min(max_det, prob.shape[1] * n_cls), dim=1)
    labels, queries = idx % n_cls, idx // n_cls
    boxes = pred_boxes.float().gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * imgsz

    records = []
    for b, meta in enumerate(metas):
        records += format_coco_predictions(
            image_id=meta["image_id"],
            boxes_letterbox=xyxy[b].cpu().numpy(),
            scores=scores[b].cpu().numpy(),
            labels=(labels[b] + 1).cpu().numpy(),
            scale_ratio=meta["scale_ratio"],
            pad=meta["pad"],
            orig_shape=meta["orig_size"],
            score_thresh=score_thresh,
        )
    return records


@torch.no_grad()
def predict(
    model: DFineForObjectDetection,
    loader: DataLoader,
    imgsz: int,
    max_det: int,
    device: Any,
    amp: bool = True,
    score_thresh: float = 0.001,
    with_loss: bool = False,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, float]]:
    """Runs the model over a loader of VisDroneDataset or FramesDataset batches.

    Returns (COCO records, mean forward time in ms per frame, mean val losses if with_loss)."""
    model.eval()
    dev = torch.device(device)
    use_amp = amp and dev.type == "cuda"
    records, total_s, n_frames = [], 0.0, 0
    loss_sums: Dict[str, float] = defaultdict(float)
    for images, targets in tqdm(loader, desc="predict", leave=False):
        pixel_values = torch.stack(images).to(dev, non_blocking=True)
        labels = _to_labels(targets, imgsz, dev) if with_loss else None
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            out = model(pixel_values=pixel_values, labels=labels)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        total_s += time.perf_counter() - t0
        n_frames += len(images)
        if with_loss:
            for k, v in _loss_summary(out.loss, out.loss_dict).items():
                loss_sums[k] += v * len(images)
        records += _postprocess(out.logits, out.pred_boxes, [_meta(t) for t in targets], imgsz, max_det, score_thresh)
    losses = {k: v / max(1, n_frames) for k, v in loss_sums.items()}
    return records, 1000.0 * total_s / max(1, n_frames), losses


def predict_frames(
    model: DFineForObjectDetection,
    frames: List[Dict[str, Any]],
    image_root: Union[str, Path],
    imgsz: int,
    batch: int,
    max_det: int,
    device: Any,
    workers: int = 4,
    amp: bool = True,
) -> Tuple[List[Dict[str, Any]], float]:
    """COCO-format predictions for arbitrary frames (e.g. the test set); returns (records, ms/frame)."""
    loader = _loader(FramesDataset(frames, image_root, imgsz), batch, workers, shuffle=False)
    records, ms, _ = predict(model, loader, imgsz, max_det, device, amp)
    return records, ms


def evaluate_dfine(
    model: DFineForObjectDetection,
    dataset_dir: Union[str, Path],
    val_stride: int,
    imgsz: int,
    batch: int,
    max_det: int,
    device: Any,
    run_dir: Optional[Union[str, Path]] = None,
    workers: int = 4,
    amp: bool = True,
) -> Dict[str, float]:
    """COCO metrics (up to 500 detections per image), speed and size on the validation frames.
    Saved to run_dir/metrics.json when run_dir is given."""
    dataset_dir = Path(dataset_dir)
    val_ds = VisDroneDataset(dataset_dir, "val", stride=val_stride, target_size=(imgsz, imgsz), normalize_imagenet=False)
    records, ms, _ = predict(model, _loader(val_ds, batch, workers, shuffle=False), imgsz, max_det, device, amp)
    coco_gt = COCO(str(dataset_dir / "val" / "annotations_coco.json"))
    metrics = evaluate_coco_metrics(coco_gt, records, image_ids=[s["id"] for s in val_ds.samples], max_dets=500)
    metrics.update({
        "ms_per_frame": ms,
        "fps": 1000.0 / ms if ms > 0 else float("nan"),
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    })
    if run_dir is not None:
        (Path(run_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# --------------------------------------------------------------------------- training


def _loss_summary(loss: torch.Tensor, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Total loss plus each loss type summed over decoder layers, encoder and denoising terms."""
    out = {"loss": float(loss.detach())}
    for t in LOSS_TYPES:
        out[t] = float(sum(v.detach() for k, v in loss_dict.items() if k.split("_")[1] == t))
    return out


def _param_groups(model: DFineForObjectDetection, lr: float, backbone_lr_mult: float, weight_decay: float):
    """Backbone / rest x decay / no decay (norms, biases and other 1-D parameters)."""
    backbone = {id(p) for p in model.model.backbone.parameters()}
    groups = defaultdict(list)
    for p in model.parameters():
        if p.requires_grad:
            groups[(id(p) in backbone, p.ndim <= 1)].append(p)
    return [
        {"params": ps, "lr": lr * (backbone_lr_mult if is_bb else 1.0), "weight_decay": 0.0 if no_wd else weight_decay}
        for (is_bb, no_wd), ps in groups.items()
    ]


def _lr_lambda(warmup_iters: int, total_iters: int, min_ratio: float):
    """Linear warm-up, then cosine decay to `min_ratio` of the base LR."""
    def f(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = min(1.0, (it - warmup_iters) / max(1, total_iters - warmup_iters))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def training_state(run_dir: Union[str, Path]) -> str:
    """Returns "new", "resume" (interrupted run) or "done" (finished run)."""
    state_file = Path(run_dir) / "train_state.json"
    if not (Path(run_dir) / "last.pt").exists() or not state_file.exists():
        return "new"
    return "done" if json.loads(state_file.read_text())["finished"] else "resume"


def train_dfine(cfg: Dict[str, Any], dataset_dir: Union[str, Path], run_dir: Union[str, Path], device: Any) -> str:
    """Trains (or resumes, or skips a finished) D-FINE run described by `cfg`; see notebook 06 for the keys.

    Each epoch: train, evaluate the EMA weights on the validation frames, append to results.csv,
    save last.pt and (on a new best mAP@[.5:.95]) best/. Returns the state before this call.
    """
    dataset_dir, run_dir, dev = Path(dataset_dir), Path(run_dir), torch.device(device)
    state = training_state(run_dir)
    print(f"Run '{run_dir.name}': {state}")
    if state == "done":
        return state
    run_dir.mkdir(parents=True, exist_ok=True)
    if state == "new":
        (run_dir / "cfg.json").write_text(json.dumps(cfg, indent=2))
    torch.manual_seed(cfg["seed"])
    imgsz = cfg["imgsz"]

    # Data
    train_ds = VisDroneDataset(
        dataset_dir, "train", stride=cfg["train_stride"], target_size=(imgsz, imgsz),
        transforms=get_train_transforms(), normalize_imagenet=False, mask_ignore=cfg["mask_ignore"],
    )
    if cfg.get("fraction", 1.0) < 1.0:
        keep = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(cfg["seed"]))
        train_ds = Subset(train_ds, sorted(keep[: int(len(train_ds) * cfg["fraction"])].tolist()))
    val_ds = VisDroneDataset(dataset_dir, "val", stride=cfg["val_stride"], target_size=(imgsz, imgsz), normalize_imagenet=False)
    train_loader = _loader(train_ds, cfg["batch"], cfg["workers"], shuffle=True, seed=cfg["seed"])
    val_loader = _loader(val_ds, cfg["batch"], cfg["workers"], shuffle=False)
    coco_gt = COCO(str(dataset_dir / "val" / "annotations_coco.json"))
    val_ids = [s["id"] for s in val_ds.samples]
    print(f"Train frames: {len(train_ds):,} | Val frames: {len(val_ds):,}")

    # Model, EMA, optimizer, schedule
    model = build_dfine(cfg["checkpoint"], cfg["class_names"], **cfg["model_config"]).to(dev)
    ema = ModelEMA(model, cfg["ema_decay"], cfg["ema_warmups"]) if cfg["ema_decay"] > 0 else None
    optimizer = torch.optim.AdamW(_param_groups(model, cfg["lr0"], cfg["backbone_lr_mult"], cfg["weight_decay"]), lr=cfg["lr0"])
    total_iters = cfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(cfg["warmup_iters"], total_iters, cfg["lr_min_ratio"])
    )
    use_amp = cfg["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_map, history = 0, -1.0, []
    if state == "resume":
        ckpt = torch.load(run_dir / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema is not None:
            ema.module.load_state_dict(ckpt["ema"])
            ema.updates = ckpt["ema_updates"]
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_map = ckpt["epoch"], ckpt["best_map"]
        history = pd.read_csv(run_dir / "results.csv").to_dict("records")
        print(f"Resuming from epoch {start_epoch + 1}/{cfg['epochs']} (best mAP {best_map:.4f})")
    # Epochs since the best mAP so far (restores the early-stopping counter on resume)
    maps = [h["metrics/mAP50-95(B)"] for h in history]
    bad_epochs = len(maps) - 1 - int(np.argmax(maps)) if maps else 0

    t_start = time.time() - (history[-1]["time"] if history else 0.0)
    for epoch in range(start_epoch, cfg["epochs"]):
        # Train one epoch
        model.train()
        sums: Dict[str, float] = defaultdict(float)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}")
        for i, (images, targets) in enumerate(pbar):
            pixel_values = torch.stack(images).to(dev, non_blocking=True)
            labels = _to_labels(targets, imgsz, dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(pixel_values=pixel_values, labels=labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(out.loss).backward()
            if cfg["clip_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            for k, v in _loss_summary(out.loss, out.loss_dict).items():
                sums[k] += v
            pbar.set_postfix(loss=f"{sums['loss'] / (i + 1):.3f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Validate (EMA weights when enabled)
        eval_model = ema.module if ema is not None else model
        records, _, val_losses = predict(eval_model, val_loader, imgsz, cfg["max_det"], dev, cfg["amp"], with_loss=True)
        m = evaluate_coco_metrics(coco_gt, records, image_ids=val_ids, max_dets=500)
        row = {
            "epoch": epoch + 1,
            "time": round(time.time() - t_start, 1),
            **{f"train/{k}": v / len(train_loader) for k, v in sums.items()},
            "metrics/mAP50(B)": m["mAP_50"],
            "metrics/mAP50-95(B)": m["mAP_50_95"],
            "metrics/mAP_small": m["mAP_small"],
            "metrics/mAR": m["mAR"],
            **{f"val/{k}": v for k, v in val_losses.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "results.csv", index=False)
        print(f"epoch {epoch + 1}: train loss {row['train/loss']:.3f} | val loss {row['val/loss']:.3f} | "
              f"mAP {m['mAP_50_95']:.4f} | mAP50 {m['mAP_50']:.4f} | AP_small {m['mAP_small']:.4f}")

        if m["mAP_50_95"] > best_map:
            best_map, bad_epochs = m["mAP_50_95"], 0
            eval_model.save_pretrained(run_dir / "best")
        else:
            bad_epochs += 1
        finished = epoch + 1 == cfg["epochs"] or bad_epochs >= cfg["patience"]

        torch.save({
            "model": model.state_dict(),
            "ema": ema.module.state_dict() if ema is not None else None,
            "ema_updates": ema.updates if ema is not None else 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "best_map": best_map,
        }, run_dir / "last.pt")
        (run_dir / "train_state.json").write_text(
            json.dumps({"epoch": epoch + 1, "best_map": best_map, "finished": finished})
        )
        if finished:
            if bad_epochs >= cfg["patience"]:
                print(f"Early stopping: no improvement for {bad_epochs} epochs")
            break
    return state


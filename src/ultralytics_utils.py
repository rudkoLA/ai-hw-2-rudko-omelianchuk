"""Glue between the shared VisDrone protocol and Ultralytics models (YOLO / RT-DETR)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from pycocotools.coco import COCO

from src.dataset import select_frames
from src.evaluation import evaluate_coco_metrics
from src.submission import build_submission, load_test_frames, validate_submission


def write_yolo_split(
    dataset_dir: Union[str, Path],
    coco_by_split: Dict[str, Dict[str, Any]],
    out_dir: Union[str, Path],
    class_names: List[str],
    stride: Union[int, Dict[str, int]] = 3,
) -> Tuple[Path, Dict[str, List[Dict[str, Any]]]]:
    """Writes Ultralytics image lists + data.yaml for the same frames the torchvision pipeline uses.

    Labels are found by Ultralytics by replacing /images/ with /labels/ in each image path.

    Args:
        dataset_dir: Competition dataset root (contains train/ and val/).
        coco_by_split: {"train": coco_dict, "val": coco_dict}.
        out_dir: Where to write <split>.txt and data.yaml.
        class_names: Class names in YOLO id order (0..7).
        stride: Temporal subsampling step (every `stride`-th frame per sequence); an int for all
            splits or a per-split dict, e.g. {"train": 15, "val": 3}.

    Returns:
        (path to data.yaml, {split: selected COCO image records}).
    """
    dataset_dir, out_dir = Path(dataset_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, List[Dict[str, Any]]] = {}
    for split, coco in coco_by_split.items():
        split_stride = stride[split] if isinstance(stride, dict) else stride
        frames[split] = select_frames(coco["images"], stride=split_stride)
        paths = [str(dataset_dir / split / img["file_name"]) for img in frames[split]]
        (out_dir / f"{split}.txt").write_text("\n".join(paths) + "\n")
        print(f"[{split}]: {len(paths):,} frames (stride={split_stride}) -> {out_dir / f'{split}.txt'}")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": str((out_dir / "train.txt").resolve()),
                "val": str((out_dir / "val.txt").resolve()),
                "nc": len(class_names),
                "names": list(class_names),
            },
            sort_keys=False,
        )
    )
    return data_yaml, frames


def training_state(run_dir: Union[str, Path]) -> str:
    """Returns "new", "resume" (interrupted run with last.pt) or "done" (finished run)."""
    last = Path(run_dir) / "weights" / "last.pt"
    if not last.exists():
        return "new"
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    # Ultralytics strips finished checkpoints: epoch=-1 and no optimizer state.
    return "done" if ckpt.get("epoch", -1) < 0 or ckpt.get("optimizer") is None else "resume"


def predict_coco(
    model: Any,
    frames: List[Dict[str, Any]],
    image_root: Union[str, Path],
    imgsz: int,
    conf: float = 0.001,
    max_det: int = 300,
    batch: int = 8,
    device: Any = None,
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Runs an Ultralytics model on image records and returns COCO-format detections.

    `frames` are dicts with "id" and "file_name" (relative to `image_root`): COCO image
    records for val, or src.submission.load_test_frames() for the test set.
    Boxes come back in original pixel coordinates; YOLO class ids (0-based) are mapped to
    COCO category ids (1-based).

    `predict_kwargs` are extra model.predict() arguments, e.g. {"iou": 0.6} (NMS IoU threshold) or
    {"nms": False} (YOLO26: NMS-free one-to-one head instead of one-to-many head + NMS).

    Returns:
        (records for evaluate_coco_metrics, mean inference time in ms per frame).
    """
    image_root = Path(image_root)
    records: List[Dict[str, Any]] = []
    total_ms = 0.0
    for start in range(0, len(frames), batch):
        chunk = frames[start : start + batch]
        results = model.predict(
            [str(image_root / img["file_name"]) for img in chunk],
            imgsz=imgsz,
            conf=conf,
            max_det=max_det,
            batch=batch,
            device=device,
            verbose=False,
            **(predict_kwargs or {}),
        )
        for img, res in zip(chunk, results):
            total_ms += res.speed["inference"]
            boxes = res.boxes
            for (x1, y1, x2, y2), score, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.int().tolist()
            ):
                records.append(
                    {
                        "image_id": img["id"],
                        "category_id": cls + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    return records, total_ms / max(1, len(frames))


def train_or_resume(
    model_cls: Any,
    source: str,
    trainer: Any,
    run_dir: Union[str, Path],
    train_args: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Starts, resumes or skips an Ultralytics training run stored in `run_dir`.

    Args:
        model_cls: Ultralytics model class (YOLO, RTDETR).
        source: Weights (.pt) or model config (.yaml) to start a new run from.
        trainer: Trainer class passed to model.train(trainer=...).
        run_dir: <project>/<name> directory of the run (last.pt / best.pt / results.csv).
        train_args: Ultralytics train() arguments for a new run (data, imgsz, epochs, ...).
        config: Optional dict saved to run_dir/cfg.json before training starts.

    Returns:
        The state before this call: "new", "resume" or "done" (finished runs are never retrained).
    """
    run_dir = Path(run_dir)
    state = training_state(run_dir)
    print(f"Run '{run_dir.name}': {state}")
    if state == "new":
        run_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (run_dir / "cfg.json").write_text(json.dumps(config, indent=2))
        model_cls(source).train(
            trainer=trainer, project=str(run_dir.parent), name=run_dir.name, exist_ok=True, **train_args
        )
    elif state == "resume":
        model_cls(str(run_dir / "weights" / "last.pt")).train(trainer=trainer, resume=True)
    return state


def plot_training_curves(run_dir: Union[str, Path]) -> pd.DataFrame:
    """Plots per-epoch validation mAP and train / val losses from results.csv; returns the table."""
    history = pd.read_csv(Path(run_dir) / "results.csv")
    history.columns = history.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for col, label in [("metrics/mAP50-95(B)", "mAP@[.5:.95]"), ("metrics/mAP50(B)", "mAP@.5")]:
        axes[0].plot(history["epoch"], history[col], marker="o", label=label)
    axes[0].set(title="Validation mAP", xlabel="epoch")
    for ax, split in zip(axes[1:], ["train", "val"]):
        for col in [c for c in history.columns if c.startswith(f"{split}/")]:
            ax.plot(history["epoch"], history[col], label=col.split("/")[1])
        ax.set(title=f"{split.capitalize()} losses", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()
    return history


def evaluate_run(
    model: Any,
    frames: List[Dict[str, Any]],
    dataset_dir: Union[str, Path],
    imgsz: int,
    max_det: int,
    batch: int,
    run_dir: Optional[Union[str, Path]] = None,
    split: str = "val",
    conf: float = 0.001,
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """COCO metrics (evaluate_coco_metrics, up to 500 detections per image), speed and size of a model
    on the given frames of `split`. Saved to run_dir/metrics.json when run_dir is given."""
    dataset_dir = Path(dataset_dir)
    records, ms_per_frame = predict_coco(
        model, frames, image_root=dataset_dir / split, imgsz=imgsz, conf=conf, max_det=max_det, batch=batch,
        predict_kwargs=predict_kwargs,
    )
    coco_gt = COCO(str(dataset_dir / split / "annotations_coco.json"))
    metrics = evaluate_coco_metrics(coco_gt, records, image_ids=[img["id"] for img in frames], max_dets=500)
    metrics.update(
        {
            "ms_per_frame": ms_per_frame,
            "fps": 1000.0 / ms_per_frame,
            "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6,
        }
    )
    if run_dir is not None:
        (Path(run_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def predict_submission(
    model: Any,
    dataset_dir: Union[str, Path],
    output_path: Union[str, Path],
    imgsz: int,
    max_det: int,
    batch: int,
    conf: float = 0.001,
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Predicts every test frame, writes the Kaggle submission CSV and returns its validation report."""
    test_frames = load_test_frames(dataset_dir)
    records, _ = predict_coco(
        model, test_frames, image_root=dataset_dir, imgsz=imgsz, conf=conf, max_det=max_det, batch=batch,
        predict_kwargs=predict_kwargs,
    )
    build_submission(records, test_frames, output_path)
    return validate_submission(output_path, test_frames)

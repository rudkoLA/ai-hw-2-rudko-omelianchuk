"""Glue between the shared VisDrone protocol and Ultralytics models (YOLO / RT-DETR)."""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import yaml

from src.dataset import select_frames


def write_yolo_split(
    dataset_dir: Union[str, Path],
    coco_by_split: Dict[str, Dict[str, Any]],
    out_dir: Union[str, Path],
    class_names: List[str],
    stride: int = 3,
) -> Tuple[Path, Dict[str, List[Dict[str, Any]]]]:
    """Writes Ultralytics image lists + data.yaml for the same frames the torchvision pipeline uses.

    Labels are found by Ultralytics by replacing /images/ with /labels/ in each image path.

    Args:
        dataset_dir: Competition dataset root (contains train/ and val/).
        coco_by_split: {"train": coco_dict, "val": coco_dict}.
        out_dir: Where to write <split>.txt and data.yaml.
        class_names: Class names in YOLO id order (0..7).
        stride: Temporal subsampling step (every `stride`-th frame per sequence).

    Returns:
        (path to data.yaml, {split: selected COCO image records}).
    """
    dataset_dir, out_dir = Path(dataset_dir).resolve(), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, List[Dict[str, Any]]] = {}
    for split, coco in coco_by_split.items():
        frames[split] = select_frames(coco["images"], stride=stride)
        paths = [str(dataset_dir / split / img["file_name"]) for img in frames[split]]
        (out_dir / f"{split}.txt").write_text("\n".join(paths) + "\n")
        print(f"[{split}]: {len(paths):,} frames (stride={stride}) -> {out_dir / f'{split}.txt'}")

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
) -> Tuple[List[Dict[str, Any]], float]:
    """Runs an Ultralytics model on COCO image records and returns COCO-format detections.

    Boxes come back in original pixel coordinates; YOLO class ids (0-based) are mapped to
    COCO category ids (1-based).

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
        )
        for img, res in zip(chunk, results):
            total_ms += res.speed["inference"]
            boxes = res.boxes
            for (x1, y1, x2, y2), score, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.int().tolist()
            ):
                records.append(
                    {
                        "image_id": int(img["id"]),
                        "category_id": cls + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    return records, total_ms / max(1, len(frames))

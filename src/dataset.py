"""Dataset and DataLoader utilities for VisDrone video object detection."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import letterbox


def select_frames(
    images: List[Dict[str, Any]],
    stride: int = 1,
    sequence_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Selects COCO image records: grouped by sequence, sorted by frame, every `stride`-th frame.

    Shared by VisDroneDataset and the Ultralytics list writer so that every
    framework trains and evaluates on exactly the same frames.
    """
    images_by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for img in images:
        vid = str(img["video"])
        if sequence_names is None or vid in sequence_names:
            images_by_seq[vid].append(img)

    samples: List[Dict[str, Any]] = []
    for vid in sorted(images_by_seq.keys()):
        frames = sorted(images_by_seq[vid], key=lambda x: int(x["frame_id"]))
        samples.extend(frames[::stride] if stride > 1 else frames)
    return samples


class VisDroneDataset(Dataset[Tuple[torch.Tensor, Dict[str, Any]]]):
    """PyTorch Dataset for VisDrone video sequences supporting GroupKFold and frame strides.

    Loads images on-the-fly, applies aspect-ratio-preserving letterboxing,
    and formats bounding boxes for PyTorch object detection models.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        sequence_names: Optional[List[str]] = None,
        stride: int = 1,
        target_size: Union[int, Tuple[int, int]] = (640, 640),
        transforms: Optional[A.Compose] = None,
        normalize_imagenet: bool = True,
        mask_ignore: bool = False,
        pre_transforms: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            dataset_dir: Root directory of competition dataset.
            split: 'train', 'val', or 'test'.
            sequence_names: Optional list of sequence names to restrict to (for GroupKFold).
            stride: Temporal subsampling step size (e.g. 3 = every 3rd frame).
            target_size: (H, W) for aspect-preserving letterboxing.
            transforms: Albumentations Compose pipeline with 'pascal_voc' bbox format.
            normalize_imagenet: If True, applies ImageNet mean & std normalization.
            mask_ignore: If True, fills ignored regions (iscrowd annotations) with gray before
                letterboxing, keeping the pixels of target boxes (see src/ignore_mask.py).
            pre_transforms: Optional callable `(img_rgb, boxes_xyxy, labels) -> (img_rgb, boxes_xyxy, labels)`
                applied to the original image (after ignore masking, before letterboxing), e.g.
                src.preprocessing.RandomZoomCrop. Training only: `scale_ratio` / `pad` then refer to the
                transformed image, while `orig_size` stays the original one.
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else tuple(target_size)
        self.transforms = transforms
        self.normalize_imagenet = normalize_imagenet
        self.mask_ignore = mask_ignore
        self.pre_transforms = pre_transforms
        self.img_dir = self.dataset_dir / split / "images"

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load COCO annotations
        coco_path = dataset_dir / split / "annotations_coco.json"
        with open(coco_path, "r") as f:
            self.coco_dict: Dict[str, Any] = json.load(f)

        # Temporally sorted, stride-subsampled frames per sequence
        self.samples: List[Dict[str, Any]] = select_frames(
            self.coco_dict["images"], stride=stride, sequence_names=sequence_names
        )

        # Pre-index target annotations by image_id
        selected_img_ids = set(img["id"] for img in self.samples)
        self.annotations_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        # Ignored regions are stored once per class as iscrowd annotations; keep unique boxes
        self.ignore_by_img: Dict[int, set] = defaultdict(set)
        for a in self.coco_dict["annotations"]:
            iid = a["image_id"]
            if iid in selected_img_ids and a.get("iscrowd", 0) == 1:
                self.ignore_by_img[iid].add(tuple(float(c) for c in a["bbox"]))
            if iid in selected_img_ids and a.get("iscrowd", 0) == 0:
                cid = int(a["category_id"])
                # Categories 1..8 are target classes (pedestrians, vehicles, cycles)
                if 1 <= cid <= 8:
                    self.annotations_by_img[iid].append(a)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_meta = self.samples[index]
        img_id = int(img_meta["id"])
        vid = str(img_meta["video"])
        fid = int(img_meta["frame_id"])

        # Load image file
        file_path = self.img_dir / vid / f"{fid:07d}.jpg"
        img_bgr = cv2.imread(str(file_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {file_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        if self.mask_ignore and self.ignore_by_img.get(img_id):
            img_rgb = self._mask_ignored(img_rgb, img_id)

        # Ground-truth boxes in original pixels [x1, y1, x2, y2]
        raw_anns = self.annotations_by_img[img_id]
        boxes_orig = np.array(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in raw_anns],
            dtype=np.float32,
        ).reshape(-1, 4)
        labels_orig = [int(a["category_id"]) for a in raw_anns]
        if self.pre_transforms is not None:
            img_rgb, boxes_orig, labels_orig = self.pre_transforms(img_rgb, boxes_orig, labels_orig)

        # Aspect-ratio preserving letterboxing
        padded_img, r, (dw, dh) = letterbox(
            img_rgb, target_shape=self.target_size, color=(114, 114, 114)
        )

        # Map ground-truth boxes to letterbox space [x1, y1, x2, y2]
        boxes_lb: List[List[float]] = []
        labels: List[int] = []

        for (bx1, by1, bx2, by2), cid in zip(boxes_orig.tolist(), labels_orig):
            x1 = bx1 * r + dw
            y1 = by1 * r + dh
            x2 = bx2 * r + dw
            y2 = by2 * r + dh

            # Clip to letterbox bounds
            x1 = max(0.0, min(float(self.target_size[1]), x1))
            y1 = max(0.0, min(float(self.target_size[0]), y1))
            x2 = max(0.0, min(float(self.target_size[1]), x2))
            y2 = max(0.0, min(float(self.target_size[0]), y2))

            # Filter degenerate zero-area boxes
            if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0:
                boxes_lb.append([x1, y1, x2, y2])
                labels.append(cid)

        # Apply Albumentations augmentations if provided
        if self.transforms is not None and len(boxes_lb) > 0:
            transformed = self.transforms(
                image=padded_img, bboxes=boxes_lb, class_labels=labels
            )
            padded_img = transformed["image"]
            boxes_lb = [list(b) for b in transformed["bboxes"]]
            labels = [int(l) for l in transformed["class_labels"]]

        # Convert image to float32 tensor (3, H, W) with normalization
        img_float = padded_img.astype(np.float32) / 255.0
        if self.normalize_imagenet:
            img_float = (img_float - self.mean) / self.std
        tensor_img = torch.from_numpy(img_float.transpose(2, 0, 1))

        # Convert targets to PyTorch tensors
        if len(boxes_lb) > 0:
            boxes_tensor = torch.as_tensor(boxes_lb, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
            iscrowd_tensor = torch.zeros((len(labels),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "scale_ratio": torch.tensor([r], dtype=torch.float32),
            "pad": torch.tensor([dw, dh], dtype=torch.float32),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return tensor_img, target

    def _mask_ignored(self, img_rgb: np.ndarray, img_id: int) -> np.ndarray:
        """Fills ignored regions with gray, keeping the pixels of target boxes visible."""
        from src.ignore_mask import mask_ignored_regions  # lazy: pulls in ultralytics

        h, w = img_rgb.shape[:2]
        ignore = _coco_to_norm_cxcywh(sorted(self.ignore_by_img[img_id]), h, w)
        targets = _coco_to_norm_cxcywh([a["bbox"] for a in self.annotations_by_img[img_id]], h, w)
        return mask_ignored_regions(img_rgb, ignore, targets)


def _coco_to_norm_cxcywh(boxes: List[Tuple[float, ...]], h: int, w: int) -> np.ndarray:
    """COCO [x, y, w, h] pixel boxes -> (N, 4) normalised cx cy w h."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2] / 2) / w, (b[:, 1] + b[:, 3] / 2) / h, b[:, 2] / w, b[:, 3] / h], axis=1)


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, Any], ...]]:
    """Custom collate function for PyTorch object detection models.

    Because each image has a variable number of bounding boxes, standard
    torch.stack fails. This collate wraps the batch into tuples.
    """
    images, targets = tuple(zip(*batch))
    return tuple(images), tuple(targets)


def build_dataloader(
    dataset: Dataset[Tuple[torch.Tensor, Dict[str, Any]]],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Constructs a production DataLoader for object detection."""
    # Only enable pinned memory on CUDA GPUs
    use_pin = pin_memory and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=use_pin,
    )


def create_visdrone_loaders(
    dataset_dir: Union[str, Path],
    target_size: Tuple[int, int] = (1024, 1024),
    batch_size: int = 4,
    val_batch_size: Optional[int] = None,
    train_stride: int = 3,
    val_stride: int = 3,
    normalize_imagenet: bool = False,
    num_workers: int = 2,
    train_seqs: Optional[List[str]] = None,
    val_seqs: Optional[List[str]] = None,
) -> Tuple[
    DataLoader[Tuple[torch.Tensor, Dict[str, Any]]],
    DataLoader[Tuple[torch.Tensor, Dict[str, Any]]],
    VisDroneDataset,
    VisDroneDataset,
]:
    """Universal factory building sequence-isolated VisDrone train and validation DataLoaders.

    Args:
        dataset_dir: Path to directory containing 'train' and 'val' subdirectories.
        target_size: (H, W) for aspect-preserving letterboxing.
        batch_size: Batch size for training loader.
        val_batch_size: Batch size for validation loader (defaults to 2x train batch size).
        train_stride: Frame subsampling stride for training sequences (k=3 recommended).
        val_stride: Frame subsampling stride for validation sequences.
        normalize_imagenet: Whether to apply ImageNet mean/std (False for torchvision detectors).
        num_workers: Number of DataLoader worker subprocesses.
        train_seqs: Optional sequence names to restrict training split to.
        val_seqs: Optional sequence names to restrict validation split to.

    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    from src.preprocessing import get_train_transforms, get_val_transforms

    ds_path = Path(dataset_dir)
    train_ds = VisDroneDataset(
        dataset_dir=ds_path,
        split="train",
        sequence_names=train_seqs,
        stride=train_stride,
        target_size=target_size,
        transforms=get_train_transforms(),
        normalize_imagenet=normalize_imagenet,
    )

    val_ds = VisDroneDataset(
        dataset_dir=ds_path,
        split="val",
        sequence_names=val_seqs,
        stride=val_stride,
        target_size=target_size,
        transforms=get_val_transforms(),
        normalize_imagenet=normalize_imagenet,
    )

    v_batch = val_batch_size if val_batch_size is not None else max(1, batch_size * 2)

    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = build_dataloader(
        val_ds,
        batch_size=v_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_ds, val_ds


def get_hardware_config(
    force_target_size: Optional[Tuple[int, int]] = None,
    force_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Dynamically resolves optimal resolution, batch size, dataloader workers,
    and precision settings based on CUDA availability and GPU VRAM capacity.

    When running in cloud GPU environments (Google Colab / Kaggle / AWS with CUDA):
      - Switches to high-resolution 1024x1024 to preserve small pedestrian features (AP_small).
      - Automatically tunes batch size based on available VRAM (e.g. L4 24GB vs. T4 16GB).
      - Enables multi-worker prefetching (num_workers=4) and pinned host memory.
      - Enables PyTorch AMP (torch.amp.autocast + GradScaler) for 3-4x Tensor Core acceleration.

    When running on local Apple Silicon (MPS) or CPU:
      - Uses 640x640 to keep unified memory consumption under 4 GB.
      - Uses batch_size=2 and num_workers=0 to avoid macOS multiprocessing socket overhead.
      - Uses standard FP32 precision.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)

        # L4 (24GB) or A100 (40/80GB) can safely support batch_size=4 at 1024x1024
        # T4 (16GB) or smaller is safer with batch_size=2 at 1024x1024 to avoid OOM
        default_batch_size = 4 if vram_gb >= 20.0 else 2
        batch_size = force_batch_size if force_batch_size is not None else default_batch_size
        target_size = force_target_size if force_target_size is not None else (1024, 1024)

        return {
            "device": torch.device("cuda"),
            "is_cuda": True,
            "target_size": target_size,
            "batch_size": batch_size,
            "val_batch_size": batch_size * 2,
            "num_workers": 4,
            "pin_memory": True,
            "use_amp": True,
            "vram_gb": vram_gb,
            "gpu_name": gpu_name,
            "description": (
                f"CUDA GPU ({gpu_name}, {vram_gb:.1f} GB VRAM) -> "
                f"High-Res {target_size[0]}x{target_size[1]}, Batch={batch_size}, "
                f"num_workers=4, Mixed Precision (AMP Enabled)"
            ),
        }
    elif torch.backends.mps.is_available():
        target_size = force_target_size if force_target_size is not None else (640, 640)
        batch_size = force_batch_size if force_batch_size is not None else 2
        return {
            "device": torch.device("mps"),
            "is_cuda": False,
            "target_size": target_size,
            "batch_size": batch_size,
            "val_batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "use_amp": False,
            "vram_gb": 0.0,
            "gpu_name": "Apple Silicon MPS",
            "description": (
                f"Apple Silicon MPS -> Lightweight {target_size[0]}x{target_size[1]}, "
                f"Batch={batch_size}, num_workers=0, FP32 (Local Development)"
            ),
        }
    else:
        target_size = force_target_size if force_target_size is not None else (640, 640)
        batch_size = force_batch_size if force_batch_size is not None else 2
        return {
            "device": torch.device("cpu"),
            "is_cuda": False,
            "target_size": target_size,
            "batch_size": batch_size,
            "val_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "use_amp": False,
            "vram_gb": 0.0,
            "gpu_name": "CPU",
            "description": (
                f"CPU Fallback -> {target_size[0]}x{target_size[1]}, "
                f"Batch={batch_size}, num_workers=0, FP32"
            ),
        }


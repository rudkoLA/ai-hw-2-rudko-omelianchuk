"""Masks VisDrone ignored regions in Ultralytics training images.

Ignored regions (crowds, dense parking lots, "others") are left out of the YOLO labels, so without
masking every unlabelled object inside them is trained as background. The scorer does not penalise
detections there, so that supervision is wrong. We paint the regions gray (114, the Ultralytics
letterbox colour) before resizing, augmentation and RAM caching, then paste back the original pixels
of labelled boxes: ~4.5% of target boxes overlap an ignored region and must stay visible.

Only the training dataset is masked. Validation and test images are left untouched, so evaluation
runs on the same unmodified frames as before.

The dataset classes live in this module (not in a notebook) so dataloader workers can pickle them.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from unittest import mock

import numpy as np
from ultralytics.data import build as ul_build
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.rtdetr import train as ul_rtdetr_train
from ultralytics.models.rtdetr.val import RTDETRDataset

FILL_VALUE = 114


def _sibling_txt(image_path: str, folder: str) -> Path:
    """<split>/images/<clip>/<frame>.jpg -> <split>/<folder>/<clip>/<frame>.txt (OS-independent)."""
    parts = list(Path(image_path).parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = folder
    return Path(*parts).with_suffix(".txt")


def _read_boxes(path: Path, skip_cols: int) -> np.ndarray:
    """Reads normalised cx cy w h boxes (after `skip_cols` leading columns) as an (N, 4) array."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.loadtxt(path, ndmin=2, dtype=np.float32)[:, skip_cols:skip_cols + 4]


def _to_pixels(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Normalised cx cy w h -> integer x1 y1 x2 y2, clipped to the image."""
    cx, cy, bw, bh = boxes.T
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
    xyxy = np.round(xyxy).astype(int)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    return xyxy


def mask_ignored_regions(
    im: np.ndarray, ignore: np.ndarray, targets: Optional[np.ndarray] = None, fill: int = FILL_VALUE
) -> np.ndarray:
    """Returns a copy of `im` with ignored regions filled, keeping pixels of target boxes intact.

    Args:
        im: HxWxC image at original resolution.
        ignore: (N, 4) normalised cx cy w h ignored regions.
        targets: (M, 4) normalised cx cy w h target boxes to keep visible (None = keep none).
        fill: Fill value for masked pixels.
    """
    if len(ignore) == 0:
        return im
    h, w = im.shape[:2]
    out = im.copy()
    for x1, y1, x2, y2 in _to_pixels(ignore, h, w):
        out[y1:y2, x1:x2] = fill
    if targets is not None:
        for x1, y1, x2, y2 in _to_pixels(targets, h, w):
            out[y1:y2, x1:x2] = im[y1:y2, x1:x2]
    return out


class IgnoreMaskMixin:
    """Dataset mixin: masks ignored regions of each source image right after it is decoded.

    `_load_source` is the earliest hook (full-resolution image, before resize / RAM cache /
    mosaic), so the mask stays aligned with the labels for every later augmentation.
    """

    def _load_source(self, i: int) -> np.ndarray:
        im = super()._load_source(i)
        f = self.im_files[i]
        ignore = _read_boxes(_sibling_txt(f, "ignore"), skip_cols=0)
        if len(ignore) == 0:
            return im
        targets = _read_boxes(_sibling_txt(f, "labels"), skip_cols=1)
        return mask_ignored_regions(im, ignore, targets)


class MaskedYOLODataset(IgnoreMaskMixin, YOLODataset):
    """YOLODataset with ignored regions masked."""


class MaskedRTDETRDataset(IgnoreMaskMixin, RTDETRDataset):
    """RTDETRDataset with ignored regions masked."""


@contextmanager
def masked_train_dataset(enabled: bool, mode: str) -> Iterator[None]:
    """While active, Ultralytics builds masked dataset classes (only when enabled and mode == "train")."""
    if not (enabled and mode == "train"):
        yield
        return
    with mock.patch.object(ul_build, "YOLODataset", MaskedYOLODataset), \
            mock.patch.object(ul_rtdetr_train, "RTDETRDataset", MaskedRTDETRDataset):
        yield

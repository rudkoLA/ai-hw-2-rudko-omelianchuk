"""Preprocessing, transformation, and validation utilities for AI Lab 2."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Tuple

import albumentations as A
import cv2
import numpy as np
from sklearn.model_selection import GroupKFold


def letterbox(
    img: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image while preserving aspect ratio (letterboxing).

    Returns:
        padded_img: Image padded to target_shape
        ratio: Scale ratio applied
        (dw, dh): Padding added to width and height (split equally on both sides)
    """
    shape = img.shape[:2]  # [height, width]
    r = min(target_shape[0] / shape[0], target_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (target_shape[1] - new_unpad[0]) / 2.0, (target_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, r, (dw, dh)


# Backward-compatible alias
letterbox_image = letterbox



BBoxFormat = Literal["coco", "pascal_voc", "albumentations", "yolo"]


def get_train_transforms(bbox_format: BBoxFormat = "pascal_voc") -> A.Compose:
    """Builds reproducible Albumentations augmentation pipeline for training.
    
    Args:
        bbox_format: Bounding box format ('pascal_voc' for [x1, y1, x2, y2], 'yolo' for [xc, yc, w, h]).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format=bbox_format,
            label_fields=["class_labels"],
            min_area=16.0,
            min_visibility=0.15,
        ),
    )


class RandomZoomCrop:
    """Crops a random window of the ORIGINAL image (before letterboxing) with the image's aspect ratio.

    Each side is scaled by s ~ U(scale), so after letterboxing objects appear up to 1 / s times larger:
    with s = 0.5 on a 2688-wide frame and a 1344-wide input, small objects are seen at native resolution.
    Boxes are clipped to the window; boxes keeping less than `min_visibility` of their area are dropped.

    Called as `(img, boxes_xyxy, labels) -> (img, boxes_xyxy, labels)` by VisDroneDataset(pre_transforms=...).
    A module-level class (not a lambda) so it pickles into DataLoader workers on Windows.
    """

    def __init__(self, scale: Tuple[float, float] = (0.5, 1.0), p: float = 0.5, min_visibility: float = 0.3) -> None:
        self.scale, self.p, self.min_visibility = tuple(scale), p, min_visibility

    def __call__(self, img: np.ndarray, boxes: np.ndarray, labels: list) -> Tuple[np.ndarray, np.ndarray, list]:
        if random.random() >= self.p:
            return img, boxes, labels
        h, w = img.shape[:2]
        s = random.uniform(*self.scale)
        cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        img = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
        if len(boxes) == 0:
            return img, boxes, labels

        shifted = boxes - np.array([x0, y0, x0, y0], dtype=boxes.dtype)
        clipped = np.clip(shifted, 0, np.array([cw, ch, cw, ch], dtype=boxes.dtype))
        area = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
        bw, bh = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
        keep = (bw >= 1) & (bh >= 1) & (bw * bh >= self.min_visibility * area)
        return img, clipped[keep], [labels[i] for i in np.flatnonzero(keep)]


def get_dfine_train_transforms(strong: bool = True, downscale_p: float = 0.0) -> A.Compose:
    """Augmentations applied to the letterboxed image for D-FINE (pascal_voc boxes).

    Compared with get_train_transforms: no rotation (rotating axis-aligned boxes makes them loose,
    which costs AP at high IoU) and min_area=1 (min_area=16 silently dropped the tiniest VisDrone objects).
    Zoom-in comes from RandomZoomCrop on the original image. `strong=False` keeps only the flip (and the
    downscale below), for the final no-augmentation epochs (as D-FINE's own `stop_epoch` policy does).

    `downscale_p`: probability of shrinking the letterboxed image to 40-80% and enlarging it back
    (INTER_AREA down, INTER_LINEAR up, the same interpolation letterbox uses). This imitates test frames
    from low-resolution sources: 66% of test frames are <= 1360 px wide (incl. 680x382 and 960x540, absent
    from training) and get upscaled by letterboxing, while training frames are mostly 1904-2688 px wide.
    It describes the test distribution rather than acting as a regulariser, so it stays on in the
    no-augmentation epochs too.
    """
    ops = [A.HorizontalFlip(p=0.5)]
    if downscale_p > 0:
        ops.append(A.Downscale(
            scale_range=(0.4, 0.8),
            interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
            p=downscale_p,
        ))
    if strong:
        ops += [
            A.Affine(scale=(0.8, 1.1), translate_percent=(-0.05, 0.05), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
        ]
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_area=1.0, min_visibility=0.3),
    )


def get_val_transforms(bbox_format: BBoxFormat = "pascal_voc") -> A.Compose:
    """Validation transformation without stochastic augmentation."""
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format=bbox_format,
            label_fields=["class_labels"],
            min_area=0.0,
            min_visibility=0.0,
        ),
    )


def split_sequences_group_kfold(
    sequence_names: list[str],
    seq_frame_counts: dict[str, int],
    n_splits: int = 5,
) -> list[tuple[list[str], list[str]]]:
    """Splits sequences using GroupKFold ensuring zero temporal overlap between folds."""
    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(len(sequence_names))
    folds = []
    for tr_idx, val_idx in gkf.split(X, groups=sequence_names):
        tr_seqs = [sequence_names[i] for i in tr_idx]
        val_seqs = [sequence_names[i] for i in val_idx]
        folds.append((tr_seqs, val_seqs))
    return folds

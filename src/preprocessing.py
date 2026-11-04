"""Preprocessing, transformation, and validation utilities for AI Lab 2."""

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

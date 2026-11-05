"""Evaluation and metrics harness for COCO-compliant video object detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def unletterbox_bboxes(
    boxes: np.ndarray,
    scale_ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int],
    to_coco_format: bool = True,
) -> np.ndarray:
    """Inverts letterbox transformation to project boxes back to original image space.

    Args:
        boxes: Array of shape (N, 4) in letterbox space [x1, y1, x2, y2].
        scale_ratio: Scaling factor applied during letterboxing.
        pad: (dw, dh) padding applied during letterboxing.
        orig_shape: (orig_h, orig_w) of unpadded raw image.
        to_coco_format: If True, converts [x1, y1, x2, y2] to COCO [x, y, w, h].

    Returns:
        Array of shape (N, 4) in original pixel coordinates.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    dw, dh = pad
    orig_h, orig_w = orig_shape

    # Invert shift and scale
    x1 = (boxes[:, 0] - dw) / scale_ratio
    y1 = (boxes[:, 1] - dh) / scale_ratio
    x2 = (boxes[:, 2] - dw) / scale_ratio
    y2 = (boxes[:, 3] - dh) / scale_ratio

    # Clip to original image boundaries
    x1 = np.clip(x1, 0.0, float(orig_w))
    y1 = np.clip(y1, 0.0, float(orig_h))
    x2 = np.clip(x2, 0.0, float(orig_w))
    y2 = np.clip(y2, 0.0, float(orig_h))

    if to_coco_format:
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        return np.column_stack([x1, y1, w, h]).astype(np.float32)

    return np.column_stack([x1, y1, x2, y2]).astype(np.float32)


def format_coco_predictions(
    image_id: int,
    boxes_letterbox: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    scale_ratio: float,
    pad: Tuple[float, float],
    orig_shape: Tuple[int, int],
    score_thresh: float = 0.05,
) -> List[Dict[str, Any]]:
    """Formats model outputs for a single image into COCO prediction records.

    Args:
        image_id: Integer COCO image ID.
        boxes_letterbox: Array (N, 4) of predicted [x1, y1, x2, y2].
        scores: Array (N,) of detection confidence scores.
        labels: Array (N,) of integer class IDs (1..8).
        scale_ratio: Scale factor from letterbox.
        pad: (dw, dh) from letterbox.
        orig_shape: (orig_h, orig_w).
        score_thresh: Minimum score to include in evaluation results.

    Returns:
        List of dicts: [{'image_id': int, 'category_id': int, 'bbox': [x,y,w,h], 'score': float}].
    """
    if len(boxes_letterbox) == 0:
        return []

    # Filter by score
    keep = scores >= score_thresh
    if not np.any(keep):
        return []

    valid_boxes = boxes_letterbox[keep]
    valid_scores = scores[keep]
    valid_labels = labels[keep]

    # Invert letterboxing to original image pixel coordinates
    coco_boxes = unletterbox_bboxes(
        valid_boxes, scale_ratio, pad, orig_shape, to_coco_format=True
    )

    records: List[Dict[str, Any]] = []
    for bbox, score, label in zip(coco_boxes, valid_scores, valid_labels):
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [float(round(c, 2)) for c in bbox],
                "score": float(round(score, 4)),
            }
        )

    return records


def evaluate_coco_metrics(
    coco_gt: COCO,
    predictions: List[Dict[str, Any]],
    image_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Runs official COCOeval benchmark and returns standardized metrics dictionary.

    Args:
        coco_gt: Initialized pycocotools COCO object for ground truth.
        predictions: List of prediction records in COCO result format.
        image_ids: Optional list of image IDs to evaluate on.

    Returns:
        Dictionary with mAP_50_95, mAP_50, mAP_75, mAP_small, mAP_medium, mAP_large, mAR_100.
    """
    if not predictions:
        return {
            "mAP_50_95": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            "mAR_100": 0.0,
        }

    # Load predictions into COCO evaluation structure
    coco_dt = coco_gt.loadRes(predictions)  # pyright: ignore[reportArgumentType]
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    if image_ids is not None:
        coco_eval.params.imgIds = image_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "mAR_100": float(stats[8]),
    }


def generate_yolo_submission(
    model: Any,
    dataset_dir: Union[str, Path],
    output_path: Union[str, Path] = "submission.csv",
    conf_thresh: float = 0.15,
    imgsz: int = 640,
    device: Optional[Union[str, int, torch.device]] = None,
    tracker_config: str = "bytetrack.yaml",
    save_raw_tracking_csv: Optional[Union[str, Path]] = "test_tracking_detections.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Runs multi-object tracking inference across all 3,811 test frames in sequence order

    grouped by video clip, resets track IDs per clip, and exports the official Kaggle
    competition submission CSV adhering to the competition specification:
    Columns: image_id, PredictionString
    Row count: Exactly 3,811 rows (one per test image in test_images.csv)
    PredictionString: Space-separated '<class> <confidence> <x> <y> <w> <h>' in pixel coordinates.
    """
    dataset_dir = Path(dataset_dir)
    test_csv = dataset_dir / "test_images.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test_images.csv not found at {test_csv}")

    df_test = pd.read_csv(test_csv)
    # Ensure chronological clip and frame ordering for tracking
    df_test_sorted = df_test.sort_values(["clip", "frame"]).reset_index(drop=True)

    det_by_image: Dict[str, List[str]] = {img_id: [] for img_id in df_test["image_id"]}
    raw_records: List[Dict[str, Any]] = []

    unique_clips = df_test_sorted["clip"].unique()
    if verbose:
        print(
            f"--> Starting test inference across {len(unique_clips)} clips ({len(df_test):,} total frames)..."
        )

    for clip_idx, clip_name in enumerate(unique_clips):
        clip_df = df_test_sorted[df_test_sorted["clip"] == clip_name]

        # Reset ByteTracker state at the start of each distinct video clip
        if (
            hasattr(model, "predictor")
            and model.predictor is not None
            and hasattr(model.predictor, "trackers")
        ):
            for t in model.predictor.trackers:
                if hasattr(t, "reset"):
                    t.reset()
                if hasattr(t, "reset_id"):
                    t.reset_id()

        clip_track_fallback = 1
        for _, item in clip_df.iterrows():
            img_id = str(item["image_id"])
            img_path = dataset_dir / item["file_name"]
            if not img_path.exists():
                continue

            results = model.track(
                source=str(img_path),
                persist=True,
                tracker=tracker_config,
                conf=conf_thresh,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            res = results[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else None
            )

            # Sort by confidence descending and cap at 500 detections per frame
            sorted_indices = np.argsort(-confs)[:500]

            orig_w = float(item["width"])
            orig_h = float(item["height"])

            for idx in sorted_indices:
                x1, y1, x2, y2 = xyxy[idx]
                c = float(confs[idx])
                cl = int(clss[idx])
                if ids is not None and ids[idx] > 0:
                    tid = int(ids[idx])
                else:
                    tid = clip_track_fallback
                    clip_track_fallback += 1

                # Original pixel coordinate clipping & width/height calculation
                x1_c = max(0.0, min(float(x1), orig_w))
                y1_c = max(0.0, min(float(y1), orig_h))
                x2_c = max(0.0, min(float(x2), orig_w))
                y2_c = max(0.0, min(float(y2), orig_h))
                w_c = max(1.0, x2_c - x1_c)
                h_c = max(1.0, y2_c - y1_c)

                # Kaggle official format: class score x y width height
                det_by_image[img_id].append(
                    f"{cl} {c:.4f} {x1_c:.1f} {y1_c:.1f} {w_c:.1f} {h_c:.1f}"
                )

                if save_raw_tracking_csv:
                    raw_records.append(
                        {
                            "image_id": img_id,
                            "clip": str(item["clip"]),
                            "frame": int(item["frame"]),
                            "track_id": int(tid),
                            "class_id": int(cl),
                            "confidence": float(round(c, 4)),
                            "x": float(round(x1_c, 2)),
                            "y": float(round(y1_c, 2)),
                            "w": float(round(w_c, 2)),
                            "h": float(round(h_c, 2)),
                        }
                    )

        if verbose and (clip_idx + 1) % 10 == 0:
            print(f"    Completed {clip_idx + 1}/{len(unique_clips)} clips...")

    # Build official Kaggle submission in exact test_images.csv order
    submission_rows = []
    empty_backfilled = 0
    for _, item in df_test.iterrows():
        img_id = str(item["image_id"])
        pred_tokens = det_by_image.get(img_id, [])
        if not pred_tokens:
            # Kaggle submission parser strictly rejects empty strings as NaN.
            # Backfill with low-confidence detection or center candidate.
            orig_w = float(item["width"])
            orig_h = float(item["height"])
            img_path = dataset_dir / item["file_name"]
            token = None
            if img_path.exists() and hasattr(model, "predict"):
                try:
                    res_fb = model.predict(
                        source=str(img_path),
                        conf=0.001,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                    )[0]
                    if len(res_fb.boxes) > 0:
                        top_idx = int(res_fb.boxes.conf.argmax().item())
                        cl_fb = int(res_fb.boxes.cls[top_idx].item())
                        c_fb = float(res_fb.boxes.conf[top_idx].item())
                        b_fb = res_fb.boxes.xyxy[top_idx].cpu().numpy()
                        x1_fb = max(0.0, float(b_fb[0]))
                        y1_fb = max(0.0, float(b_fb[1]))
                        w_fb = max(1.0, min(float(b_fb[2]), orig_w) - x1_fb)
                        h_fb = max(1.0, min(float(b_fb[3]), orig_h) - y1_fb)
                        token = f"{cl_fb} {c_fb:.4f} {x1_fb:.1f} {y1_fb:.1f} {w_fb:.1f} {h_fb:.1f}"
                except Exception:
                    token = None
            if not token:
                token = f"3 0.0100 {orig_w / 2:.1f} {orig_h / 2:.1f} 50.0 50.0"
            pred_tokens = [token]
            empty_backfilled += 1

        submission_rows.append(
            {
                "image_id": img_id,
                "PredictionString": " ".join(pred_tokens),
            }
        )

    if empty_backfilled > 0 and verbose:
        print(
            f"--> Automatically backfilled {empty_backfilled} zero-detection frames to guarantee zero nulls."
        )

    sub_df = pd.DataFrame(submission_rows)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)

    if save_raw_tracking_csv and raw_records:
        raw_path = Path(save_raw_tracking_csv)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(raw_records).to_csv(raw_path, index=False)
        if verbose:
            print(f"--> Saved detailed tracking table to {raw_path}")

    if verbose:
        print(
            f"--> Successfully generated official Kaggle submission at {out_path} "
            f"({len(sub_df):,} rows matching test set, 0 duplicates)"
        )

    return sub_df


def validate_submission_file(
    submission_path: Union[str, Path],
    dataset_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Validates that a submission CSV adheres strictly to competition requirements:

    - Exact columns: ['image_id', 'PredictionString']
    - Exactly 3,811 unique rows matching test_images.csv
    - Zero duplicate image_id entries
    - Valid token structure: groups of 6 (class score x y width height)
    """
    sub_path = Path(submission_path)
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file does not exist at {sub_path}")

    df = pd.read_csv(sub_path)
    expected_cols = ["image_id", "PredictionString"]

    col_match = list(df.columns) == expected_cols
    total_rows = len(df)
    unique_ids = int(df["image_id"].nunique()) if "image_id" in df.columns else 0
    duplicate_count = total_rows - unique_ids

    # Check nulls in image_id and PredictionString
    null_ids = int(df["image_id"].isnull().sum()) if "image_id" in df.columns else total_rows
    null_preds = int(df["PredictionString"].isnull().sum()) if "PredictionString" in df.columns else total_rows

    # Check prediction formatting
    valid_syntax = True
    images_with_detections = 0
    total_detections = 0

    if col_match:
        for val in df["PredictionString"].dropna():
            s = str(val).strip()
            if not s:
                continue
            images_with_detections += 1
            tokens = s.split()
            if len(tokens) % 6 != 0:
                valid_syntax = False
                break
            num_dets = len(tokens) // 6
            total_detections += num_dets
            if num_dets > 500:
                valid_syntax = False
                break

    is_valid = (
        col_match
        and total_rows == 3811
        and duplicate_count == 0
        and null_ids == 0
        and null_preds == 0
        and valid_syntax
        and images_with_detections > 0
    )

    report = {
        "file": str(sub_path),
        "total_rows": total_rows,
        "expected_rows": 3811,
        "unique_images": unique_ids,
        "duplicate_ids": duplicate_count,
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "columns_valid": col_match,
        "syntax_valid": valid_syntax,
        "is_valid": is_valid,
    }
    return report



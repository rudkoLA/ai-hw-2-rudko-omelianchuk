"""Two-stage anchor-based object detector (Faster R-CNN) training and evaluation harness."""

from collections import defaultdict
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO
import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.dataset import get_hardware_config
from src.evaluation import evaluate_coco_metrics, format_coco_predictions


def build_faster_rcnn_v2(
    num_classes: int = 9,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    box_positive_fraction: float = 0.25,
    box_batch_size_per_image: int = 512,
    trainable_backbone_layers: int = 3,
    pretrained: bool = True,
) -> FasterRCNN:
    """Builds Faster R-CNN with ResNet-50-FPN V2 backbone and custom two-stage hyperparameters.

    Args:
        num_classes: Number of classes including background (8 targets + 1 background = 9).
        rpn_post_nms_top_n_train: Number of RPN proposals passed to RoI head during training.
        rpn_post_nms_top_n_test: Number of RPN proposals passed to RoI head during testing.
        box_positive_fraction: Fraction of positive (foreground) RoI samples in Stage 2.
        box_batch_size_per_image: Total number of RoI samples per image in Stage 2.
        trainable_backbone_layers: Number of trainable backbone blocks (3 = fine-tuning top layers).
        pretrained: If True, initializes with COCO pre-trained weights.

    Returns:
        Configured FasterRCNN model.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

    # Instantiate base model
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # 1. Adapt RPN proposal parameters (Parameter 1)
    rpn: Any = model.rpn
    rpn._post_nms_top_n["training"] = rpn_post_nms_top_n_train
    rpn._post_nms_top_n["testing"] = rpn_post_nms_top_n_test

    # 2. Adapt RoI sampling parameters (Parameter 2)
    roi_heads: Any = model.roi_heads
    roi_heads.fg_bg_sampler.positive_fraction = box_positive_fraction
    roi_heads.fg_bg_sampler.batch_size_per_image = box_batch_size_per_image

    # 3. Replace Stage 2 box predictor head for VisDrone classes (8 + 1 background)
    in_features = roi_heads.box_predictor.cls_score.in_features
    roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Any,
    device: Any,
    scaler: Optional[torch.amp.GradScaler] = None,  # pyright: ignore[reportPrivateImportUsage]
    max_batches: Optional[int] = None,
    print_freq: int = 10,
) -> Dict[str, float]:
    """Trains detector for a single epoch and tracks multi-task loss components."""
    model.train()
    running_losses: Dict[str, float] = defaultdict(float)
    batch_count = 0
    t0 = time.time()
    dev = torch.device(device)

    for i, (images, targets) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break

        images = [img.to(dev) for img in images]
        targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Mixed precision training
        use_amp = scaler is not None and dev.type == "cuda"
        with torch.amp.autocast(device_type=dev.type, enabled=use_amp):
            loss_dict: Dict[str, torch.Tensor] = model(images, targets)
            losses: torch.Tensor = torch.stack(list(loss_dict.values())).sum()

        if scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        batch_count += 1
        for k, v in loss_dict.items():
            running_losses[k] += float(v.item())
        running_losses["loss_total"] += float(losses.item())

        if (i + 1) % print_freq == 0 or (i + 1) == len(data_loader):
            elapsed = time.time() - t0
            print(
                f"  Batch [{i+1:4d}/{len(data_loader):4d}] | "
                f"Loss: {losses.item():.4f} (Cls: {loss_dict['loss_classifier'].item():.3f}, "
                f"Box: {loss_dict['loss_box_reg'].item():.3f}, "
                f"RPN_Obj: {loss_dict['loss_objectness'].item():.3f}, "
                f"RPN_Box: {loss_dict['loss_rpn_box_reg'].item():.3f}) | "
                f"Speed: {batch_count / elapsed:.2f} batch/s"
            )

    avg_losses = {k: v / max(1, batch_count) for k, v in running_losses.items()}
    return avg_losses


def evaluate_model_performance(
    model: nn.Module,
    data_loader: Any,
    coco_gt: COCO,
    device: Any,
    max_batches: Optional[int] = None,
    score_thresh: float = 0.05,
) -> Dict[str, Any]:
    """Runs validation evaluation, measures inference latency, and computes COCO mAP."""
    model.eval()
    dev = torch.device(device)
    all_predictions: List[Dict[str, Any]] = []
    eval_image_ids: List[int] = []

    total_inference_time = 0.0
    total_frames = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break

            images_dev = [img.to(dev) for img in images]

            # Measure raw inference latency
            t_start = time.time()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            elif dev.type == "mps":
                torch.mps.synchronize()

            preds = model(images_dev)

            if dev.type == "cuda":
                torch.cuda.synchronize()
            elif dev.type == "mps":
                torch.mps.synchronize()
            t_batch = time.time() - t_start

            total_inference_time += t_batch
            total_frames += len(images)

            for target, pred in zip(targets, preds):
                iid = int(target["image_id"][0])
                eval_image_ids.append(iid)

                recs = format_coco_predictions(
                    image_id=iid,
                    boxes_letterbox=pred["boxes"].cpu().numpy(),
                    scores=pred["scores"].cpu().numpy(),
                    labels=pred["labels"].cpu().numpy(),
                    scale_ratio=float(target["scale_ratio"][0]),
                    pad=(float(target["pad"][0]), float(target["pad"][1])),
                    orig_shape=(int(target["orig_size"][0]), int(target["orig_size"][1])),
                    score_thresh=score_thresh,
                )
                all_predictions.extend(recs)

    # Compute COCO metrics
    coco_metrics = evaluate_coco_metrics(coco_gt, all_predictions, image_ids=eval_image_ids)

    # Compute inference speed
    fps = total_frames / total_inference_time if total_inference_time > 0 else 0.0
    latency_ms = (total_inference_time / total_frames * 1000.0) if total_frames > 0 else 0.0

    return {
        **coco_metrics,
        "inference_fps": float(fps),
        "latency_ms_per_frame": float(latency_ms),
        "evaluated_frames": total_frames,
        "total_detections": len(all_predictions),
    }


def analyze_failure_cases(
    model: nn.Module,
    data_loader: Any,
    device: Any,
    iou_thresh: float = 0.50,
    score_thresh: float = 0.30,
    max_batches: Optional[int] = 10,
) -> Dict[str, Any]:
    """Identifies and extracts representative failure cases:

    1. False Negatives (Missed small/occluded objects)
    2. False Positives (Spurious background detections)
    3. Classification Confusion / Localization errors
    """
    model.eval()
    dev = torch.device(device)

    cases: Dict[str, List[Dict[str, Any]]] = {
        "false_negatives": [],
        "false_positives": [],
        "classification_errors": [],
    }

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images_dev = [img.to(dev) for img in images]
            preds = model(images_dev)

            for img, target, pred in zip(images, targets, preds):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                p_boxes = pred["boxes"].cpu().numpy()
                p_scores = pred["scores"].cpu().numpy()
                p_labels = pred["labels"].cpu().numpy()

                keep = p_scores >= score_thresh
                p_boxes = p_boxes[keep]
                p_scores = p_scores[keep]
                p_labels = p_labels[keep]

                # 1. Check for False Negatives (GT boxes with no overlap)
                for gb, gl in zip(gt_boxes, gt_labels):
                    if len(p_boxes) == 0:
                        cases["false_negatives"].append({
                            "gt_box": gb,
                            "gt_label": int(gl),
                            "best_iou": 0.0,
                            "area": float((gb[2] - gb[0]) * (gb[3] - gb[1])),
                        })
                        continue

                    # Compute IoU with all predictions
                    ix1 = np.maximum(gb[0], p_boxes[:, 0])
                    iy1 = np.maximum(gb[1], p_boxes[:, 1])
                    ix2 = np.minimum(gb[2], p_boxes[:, 2])
                    iy2 = np.minimum(gb[3], p_boxes[:, 3])
                    iw = np.maximum(0.0, ix2 - ix1)
                    ih = np.maximum(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (gb[2] - gb[0]) * (gb[3] - gb[1]) + (p_boxes[:, 2] - p_boxes[:, 0]) * (p_boxes[:, 3] - p_boxes[:, 1]) - inter
                    ious = np.where(union > 0, inter / union, 0.0)
                    best_iou = float(np.max(ious)) if len(ious) > 0 else 0.0

                    if best_iou < iou_thresh:
                        cases["false_negatives"].append({
                            "gt_box": gb,
                            "gt_label": int(gl),
                            "best_iou": best_iou,
                            "area": float((gb[2] - gb[0]) * (gb[3] - gb[1])),
                        })

                # 2. Check for False Positives (predictions with no GT overlap)
                for pb, ps, pl in zip(p_boxes, p_scores, p_labels):
                    if len(gt_boxes) == 0:
                        cases["false_positives"].append({
                            "pred_box": pb,
                            "pred_label": int(pl),
                            "score": float(ps),
                            "best_iou": 0.0,
                        })
                        continue

                    ix1 = np.maximum(pb[0], gt_boxes[:, 0])
                    iy1 = np.maximum(pb[1], gt_boxes[:, 1])
                    ix2 = np.minimum(pb[2], gt_boxes[:, 2])
                    iy2 = np.minimum(pb[3], gt_boxes[:, 3])
                    iw = np.maximum(0.0, ix2 - ix1)
                    ih = np.maximum(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (pb[2] - pb[0]) * (pb[3] - pb[1]) + (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) - inter
                    ious = np.where(union > 0, inter / union, 0.0)
                    best_iou = float(np.max(ious)) if len(ious) > 0 else 0.0

                    if best_iou < 0.30:
                        cases["false_positives"].append({
                            "pred_box": pb,
                            "pred_label": int(pl),
                            "score": float(ps),
                            "best_iou": best_iou,
                        })

                # 3. Check for Classification / Localization confusion
                for pb, ps, pl in zip(p_boxes, p_scores, p_labels):
                    for gb, gl in zip(gt_boxes, gt_labels):
                        ix1 = max(pb[0], gb[0])
                        iy1 = max(pb[1], gb[1])
                        ix2 = min(pb[2], gb[2])
                        iy2 = min(pb[3], gb[3])
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        union = (pb[2] - pb[0]) * (pb[3] - pb[1]) + (gb[2] - gb[0]) * (gb[3] - gb[1]) - inter
                        iou = inter / union if union > 0 else 0.0

                        if iou >= 0.40 and pl != gl:
                            cases["classification_errors"].append({
                                "pred_box": pb,
                                "pred_label": int(pl),
                                "gt_box": gb,
                                "gt_label": int(gl),
                                "score": float(ps),
                                "iou": float(iou),
                            })

            if (
                len(cases["false_negatives"]) >= 5
                and len(cases["false_positives"]) >= 5
                and len(cases["classification_errors"]) >= 3
            ):
                break

    return cases

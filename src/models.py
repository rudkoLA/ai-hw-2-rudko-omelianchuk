"""Object detector architectures (Faster R-CNN, RetinaNet), training loops,
validation evaluation, and failure case diagnostics.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches
from pycocotools.coco import COCO
from torch import nn
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from torchvision.ops import sigmoid_focal_loss

from src.evaluation import evaluate_coco_metrics, format_coco_predictions


def build_faster_rcnn_v2(
    num_classes: int = 9,
    anchor_sizes: tuple[tuple[int, ...], ...] | None = None,
    aspect_ratios: tuple[tuple[float, ...], ...] | None = None,
    rpn_pre_nms_top_n_train: int | None = None,
    rpn_pre_nms_top_n_test: int | None = None,
    rpn_post_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_test: int = 1000,
    box_positive_fraction: float = 0.25,
    box_batch_size_per_image: int = 512,
    trainable_backbone_layers: int = 3,
    min_size: int = 1024,
    max_size: int = 1024,
    box_detections_per_img: int = 500,
    box_loss_beta: float = 1 / 9,
    label_smoothing: float = 0.0,
    pretrained: bool = True,
) -> FasterRCNN:
    """Builds Faster R-CNN with ResNet-50-FPN V2 backbone and custom two-stage hyperparameters.

    Args:
        num_classes: Number of classes including background (8 targets + 1 background = 9).
        anchor_sizes: Base anchor sizes per FPN feature level ((P2,), (P3,), (P4,), (P5,), (P6,)).
        aspect_ratios: Aspect ratios per feature level.
        rpn_pre_nms_top_n_train: Pre-NMS proposal candidates during training.
        rpn_pre_nms_top_n_test: Pre-NMS proposal candidates during testing.
        rpn_post_nms_top_n_train: Number of RPN proposals passed to RoI head during training.
        rpn_post_nms_top_n_test: Number of RPN proposals passed to RoI head during testing.
        box_positive_fraction: Fraction of positive (foreground) RoI samples in Stage 2.
        box_batch_size_per_image: Total number of RoI samples per image in Stage 2.
        trainable_backbone_layers: Number of trainable backbone blocks (3 = fine-tuning top layers).
        min_size: Minimum image size for internal transform (1024 prevents downscaling).
        max_size: Maximum image size for internal transform (1024 maintains aspect scale).
        box_detections_per_img: Max detections per image (500 accommodates dense VisDrone scenes).
        box_loss_beta: Smooth L1 transition threshold beta (tuning to e.g. 0.02 penalizes small errors linearly).
        label_smoothing: Label smoothing factor applied to RoI classification cross-entropy.
        pretrained: If True, initializes with official COCO pre-trained weights.

    Returns:
        Configured FasterRCNN model.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

    # Instantiate base model with full resolution and detection cap
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=(min_size,),
        max_size=max_size,
        box_detections_per_img=box_detections_per_img,
    )

    # 0. Custom Anchor Generator for small drone targets
    if anchor_sizes is not None and aspect_ratios is not None:
        model.rpn.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # 1. Adapt RPN proposal parameters (Parameter 1)
    rpn: Any = model.rpn
    rpn._post_nms_top_n["training"] = rpn_post_nms_top_n_train
    rpn._post_nms_top_n["testing"] = rpn_post_nms_top_n_test
    if rpn_pre_nms_top_n_train is not None:
        rpn._pre_nms_top_n["training"] = rpn_pre_nms_top_n_train
    if rpn_pre_nms_top_n_test is not None:
        rpn._pre_nms_top_n["testing"] = rpn_pre_nms_top_n_test

    # 2. Adapt RoI sampling parameters (Parameter 2)
    roi_heads: Any = model.roi_heads
    roi_heads.fg_bg_sampler.positive_fraction = box_positive_fraction
    roi_heads.fg_bg_sampler.batch_size_per_image = box_batch_size_per_image

    # 3. Replace Stage 2 box predictor head for VisDrone classes (8 + 1 background)
    in_features = roi_heads.box_predictor.cls_score.in_features
    roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 4. Loss modification: tune regression beta and/or classification label smoothing
    if abs(box_loss_beta - (1.0 / 9.0)) > 1e-5 or label_smoothing > 0.0:
        import types

        import torch.nn.functional as F

        def custom_roi_forward(self: Any, features: Any, proposals: Any, image_shapes: Any, targets: Any = None) -> Any:
            if targets is not None:
                for t in targets:
                    if t["boxes"].dtype not in (torch.float, torch.double, torch.half):
                        raise TypeError(f"target boxes must be float, got {t['boxes'].dtype}")
                    if not t["labels"].dtype == torch.int64:
                        raise TypeError(f"target labels must be int64, got {t['labels'].dtype}")

            if self.training:
                proposals, _matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            else:
                labels, regression_targets, _matched_idxs = None, None, None

            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)

            result: list[dict[str, torch.Tensor]] = []
            losses: dict[str, torch.Tensor] = {}
            if self.training:
                if labels is None or regression_targets is None:
                    raise ValueError("labels and regression_targets cannot be None in training")
                cat_labels = torch.cat(labels, dim=0)
                cat_reg_targets = torch.cat(regression_targets, dim=0)
                cls_loss = F.cross_entropy(class_logits, cat_labels, label_smoothing=label_smoothing)
                sampled_pos_inds_subset = torch.where(cat_labels > 0)[0]
                labels_pos = cat_labels[sampled_pos_inds_subset]
                N, _ = class_logits.shape
                box_regression_reshaped = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
                reg_loss = F.smooth_l1_loss(
                    box_regression_reshaped[sampled_pos_inds_subset, labels_pos],
                    cat_reg_targets[sampled_pos_inds_subset],
                    beta=box_loss_beta,
                    reduction="sum",
                ) / cat_labels.numel()
                losses = {"loss_classifier": cls_loss, "loss_box_reg": reg_loss}
            else:
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
                for i in range(len(boxes)):
                    result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i]})
            return result, losses

        roi_heads.forward = types.MethodType(custom_roi_forward, roi_heads)

    return model


class CustomRetinaNetClassificationHead(RetinaNetClassificationHead):
    """RetinaNet classification head supporting configurable Focal Loss alpha and gamma."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        norm_layer: Any | None = None,
    ) -> None:
        super().__init__(in_channels, num_anchors, num_classes, norm_layer=norm_layer)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(
            targets, cls_logits, matched_idxs
        ):
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][
                    matched_idxs_per_image[foreground_idxs_per_image]
                ],
            ] = 1.0

            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction="sum",
                )
                / max(1, int(num_foreground.item()))
            )

        return torch.stack(losses).sum() / max(1, len(targets))


def build_retinanet_v2(
    num_classes: int = 9,
    anchor_sizes: tuple[tuple[int, ...], ...] | None = None,
    aspect_ratios: tuple[tuple[float, ...], ...] | None = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    iou_high_threshold: float = 0.5,
    iou_low_threshold: float = 0.4,
    min_size: int = 1024,
    max_size: int = 1024,
    detections_per_img: int = 500,
    topk_candidates: int = 2000,
    trainable_backbone_layers: int = 3,
    pretrained: bool = True,
) -> RetinaNet:
    """Builds RetinaNet with ResNet-50-FPN V2 backbone, custom anchors, and focal loss parameters.

    Args:
        num_classes: Number of classes (8 VisDrone targets + 1 = 9).
        anchor_sizes: Multi-scale base anchor sizes per FPN level ((P3), (P4), (P5), (P6), (P7)).
                      If None, uses torchvision defaults ((32,), (64,), (128,), (256,), (512,)).
        aspect_ratios: Aspect ratios per scale ((r1, r2, ...),) per FPN level.
        focal_alpha: Focal Loss alpha weighting factor (balance between positive and negative).
        focal_gamma: Focal Loss focusing parameter (down-weights easy examples).
        iou_high_threshold: IoU threshold above which an anchor is matched as positive.
        iou_low_threshold: IoU threshold below which an anchor is matched as negative.
        min_size: Minimum image size for internal transform.
        max_size: Maximum image size for internal transform.
        detections_per_img: Maximum detections retained per image.
        topk_candidates: Pre-NMS candidate proposals per feature level.
        trainable_backbone_layers: Trainable backbone stages (3 = freeze lower ResNet).
        pretrained: If True, initializes with official COCO pre-trained weights.

    Returns:
        Configured RetinaNet model.
    """
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None

    model = retinanet_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=(min_size,),
        max_size=max_size,
        detections_per_img=detections_per_img,
        topk_candidates=topk_candidates,
    )

    # 1. Custom Anchor Generator (grounded in EDA object sizes < 32 px)
    if anchor_sizes is not None and aspect_ratios is not None:
        model.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )

    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    # 2. Matcher IoU thresholds
    model.proposal_matcher.high_threshold = iou_high_threshold
    model.proposal_matcher.low_threshold = iou_low_threshold

    # 3. Replace classification and regression heads for VisDrone classes & anchor count
    in_channels = model.backbone.out_channels
    model.head.classification_head = CustomRetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )
    model.head.regression_head = RetinaNetRegressionHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
    )

    return model


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Any,
    device: Any,
    scaler: torch.amp.GradScaler | None = None,  # pyright: ignore[reportPrivateImportUsage]
    max_batches: int | None = None,
    print_freq: int = 20,
    grad_clip_norm: float = 10.0,
) -> dict[str, float]:
    """Trains detector for a single epoch and tracks multi-task loss components."""
    model.train()
    running_losses: dict[str, float] = defaultdict(float)
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
            loss_dict: dict[str, torch.Tensor] = model(images, targets)
            losses: torch.Tensor = torch.stack(list(loss_dict.values())).sum()

        if not torch.isfinite(losses):
            print(f"  [Warning] Non-finite loss at batch {i+1}: {loss_dict}. Skipping batch.")
            optimizer.zero_grad()
            continue

        if scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        batch_count += 1
        for k, v in loss_dict.items():
            running_losses[k] += float(v.item())
        running_losses["loss_total"] += float(losses.item())

        if (i + 1) % print_freq == 0 or (i + 1) == len(data_loader):
            elapsed = time.time() - t0
            loss_str = " | ".join(f"{k}: {v.item():.3f}" for k, v in loss_dict.items())
            print(
                f"  Batch [{i+1:4d}/{len(data_loader):4d}] | "
                f"Total: {losses.item():.4f} ({loss_str}) | "
                f"Speed: {batch_count / elapsed:.2f} batch/s"
            )

    avg_losses = {k: v / max(1, batch_count) for k, v in running_losses.items()}
    return avg_losses


def train_detector_model(
    model: nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: Any,
    num_epochs: int = 5,
    lr_scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,  # pyright: ignore[reportPrivateImportUsage]
    checkpoint_dir: str | Path = Path("checkpoints"),
    model_tag: str = "detector",
    print_freq: int = 25,
    max_batches: int | None = None,
    grad_clip_norm: float = 10.0,
    resume_checkpoint: str | Path | None = None,
    save_every_epoch: bool = False,
) -> list[dict[str, Any]]:
    """Standardized multi-epoch detector training loop with checkpoint persistence and auto-resume."""
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    start_epoch = 0
    if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        print(f"--> Loading resume checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (KeyError, ValueError, RuntimeError) as e_opt:
                    print(f"  [Notice] Optimizer state could not be loaded: {e_opt}. Using initialized optimizer.")
            start_epoch = int(ckpt.get("epoch", 0))
            if lr_scheduler is not None and "scheduler_state_dict" in ckpt:
                try:
                    lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    print("--> Successfully restored learning rate scheduler state.")
                except (KeyError, ValueError, RuntimeError) as e_sched:
                    print(f"  [Notice] Scheduler state could not be loaded: {e_sched}.")
            elif lr_scheduler is not None and start_epoch > 0:
                for _ in range(start_epoch):
                    lr_scheduler.step()
                print(f"--> Fast-forwarded learning rate scheduler to epoch {start_epoch}.")

            if "history" in ckpt and isinstance(ckpt["history"], list):
                history = list(ckpt["history"])
            elif (ckpt_path / f"{model_tag}_history.json").exists():
                try:
                    with open(ckpt_path / f"{model_tag}_history.json") as f_h:
                        history = json.load(f_h)
                except (OSError, json.JSONDecodeError) as e_json:
                    print(f"  [Notice] Could not load history file: {e_json}")
            print(f"--> Successfully resumed model at epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)
            print("--> Loaded model weights directly from state dict.")

    if start_epoch >= num_epochs:
        print(f"--> Target epochs ({num_epochs}) already completed (current: {start_epoch}). Skipping training.")
        return history

    print(f"--> Starting training: Epoch [{start_epoch + 1}/{num_epochs}] for '{model_tag}' on {device}")
    total_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        t_ep0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print("=" * 60)
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Learning Rate: {current_lr:.6f}")
        print("=" * 60)

        epoch_losses = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            scaler=scaler,
            max_batches=max_batches,
            print_freq=print_freq,
            grad_clip_norm=grad_clip_norm,
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        ep_duration = time.time() - t_ep0
        epoch_losses["epoch"] = epoch + 1
        epoch_losses["lr"] = current_lr
        epoch_losses["duration_s"] = ep_duration
        history.append(epoch_losses)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] finished in {ep_duration / 60:.1f}m ({ep_duration:.1f}s) | "
            f"Total Loss: {epoch_losses['loss_total']:.4f}"
        )

        # Save latest checkpoint
        ckpt_payload: dict[str, Any] = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_losses": epoch_losses,
            "history": history,
        }
        if lr_scheduler is not None:
            ckpt_payload["scheduler_state_dict"] = lr_scheduler.state_dict()
        torch.save(ckpt_payload, ckpt_path / f"{model_tag}_last.pt")
        if save_every_epoch:
            torch.save(ckpt_payload, ckpt_path / f"{model_tag}_epoch_{epoch + 1}.pt")

        # Save training history JSON
        with open(ckpt_path / f"{model_tag}_history.json", "w") as f_json:
            json.dump(history, f_json, indent=2)

    total_time = time.time() - total_start
    print(f"--> Training completed in {total_time / 60:.1f} minutes. Saved to: {ckpt_path}")
    return history


def evaluate_detector_model(
    model: nn.Module,
    data_loader: Any,
    coco_gt: COCO,
    device: Any,
    score_thresh: float = 0.05,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Runs validation evaluation, measures latency/FPS, and computes COCO mAP."""
    model.eval()
    dev = torch.device(device)
    all_predictions: list[dict[str, Any]] = []
    eval_image_ids: list[int] = []

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

    # Compute official COCO metrics
    coco_metrics = evaluate_coco_metrics(coco_gt, all_predictions, image_ids=eval_image_ids)

    # Compute inference latency & throughput
    fps = total_frames / total_inference_time if total_inference_time > 0 else 0.0
    latency_ms = (total_inference_time / total_frames * 1000.0) if total_frames > 0 else 0.0

    return {
        **coco_metrics,
        "inference_fps": float(fps),
        "latency_ms_per_frame": float(latency_ms),
        "evaluated_frames": total_frames,
        "total_detections": len(all_predictions),
    }


# Backward-compatible alias
evaluate_model_performance = evaluate_detector_model



def analyze_failure_cases(
    model: nn.Module,
    data_loader: Any,
    device: Any,
    iou_thresh: float = 0.50,
    score_thresh: float = 0.30,
    max_batches: int | None = 10,
) -> dict[str, Any]:
    """Identifies representative failure cases:
    1. False Negatives (missed small/occluded objects)
    2. False Positives (spurious background detections)
    3. Classification Confusion / Localization errors
    """
    model.eval()
    dev = torch.device(device)

    cases: dict[str, list[dict[str, Any]]] = {
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

                # 1. False Negatives
                for gb, gl in zip(gt_boxes, gt_labels):
                    if len(p_boxes) == 0:
                        cases["false_negatives"].append({
                            "gt_box": gb,
                            "gt_label": int(gl),
                            "best_iou": 0.0,
                            "area": float((gb[2] - gb[0]) * (gb[3] - gb[1])),
                            "image": img.cpu(),
                        })
                        continue

                    ix1 = np.maximum(gb[0], p_boxes[:, 0])
                    iy1 = np.maximum(gb[1], p_boxes[:, 1])
                    ix2 = np.minimum(gb[2], p_boxes[:, 2])
                    iy2 = np.minimum(gb[3], p_boxes[:, 3])
                    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                    union = (gb[2] - gb[0]) * (gb[3] - gb[1]) + (p_boxes[:, 2] - p_boxes[:, 0]) * (p_boxes[:, 3] - p_boxes[:, 1]) - inter
                    ious = np.where(union > 0, inter / union, 0.0)
                    best_iou = float(np.max(ious)) if len(ious) > 0 else 0.0

                    if best_iou < iou_thresh:
                        cases["false_negatives"].append({
                            "gt_box": gb,
                            "gt_label": int(gl),
                            "best_iou": best_iou,
                            "area": float((gb[2] - gb[0]) * (gb[3] - gb[1])),
                            "image": img.cpu(),
                        })

                # 2. False Positives
                for pb, ps, pl in zip(p_boxes, p_scores, p_labels):
                    if len(gt_boxes) == 0:
                        cases["false_positives"].append({
                            "pred_box": pb,
                            "pred_label": int(pl),
                            "score": float(ps),
                            "best_iou": 0.0,
                            "image": img.cpu(),
                        })
                        continue

                    ix1 = np.maximum(pb[0], gt_boxes[:, 0])
                    iy1 = np.maximum(pb[1], gt_boxes[:, 1])
                    ix2 = np.minimum(pb[2], gt_boxes[:, 2])
                    iy2 = np.minimum(pb[3], gt_boxes[:, 3])
                    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                    union = (pb[2] - pb[0]) * (pb[3] - pb[1]) + (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) - inter
                    ious = np.where(union > 0, inter / union, 0.0)
                    best_iou = float(np.max(ious)) if len(ious) > 0 else 0.0

                    if best_iou < 0.20:
                        cases["false_positives"].append({
                            "pred_box": pb,
                            "pred_label": int(pl),
                            "score": float(ps),
                            "best_iou": best_iou,
                            "image": img.cpu(),
                        })

                # 3. Classification confusion
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
                                "image": img.cpu(),
                            })

            if (
                len(cases["false_negatives"]) >= 5
                and len(cases["false_positives"]) >= 5
                and len(cases["classification_errors"]) >= 3
            ):
                break

    return cases


def plot_failure_cases(
    cases: dict[str, list[dict[str, Any]]],
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plots visual diagnostic panels for the three failure case categories."""
    cats = class_names or [
        "ignored", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"
    ]

    _fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

    # 1. False Negative
    ax0 = axes[0]
    if cases["false_negatives"]:
        fn = cases["false_negatives"][0]
        img = fn["image"].permute(1, 2, 0).numpy().clip(0, 1)
        ax0.imshow(img)
        gb = fn["gt_box"]
        rect = patches.Rectangle(
            (gb[0], gb[1]), gb[2] - gb[0], gb[3] - gb[1],
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax0.add_patch(rect)
        cname = cats[fn["gt_label"]] if fn["gt_label"] < len(cats) else f"cls_{fn['gt_label']}"
        ax0.set_title(f"False Negative (Missed Target)\nGT: {cname} (Area={fn['area']:.0f}px)", color="darkgreen", weight="bold")
    else:
        ax0.text(0.5, 0.5, "No False Negatives extracted", ha="center")
    ax0.axis("off")

    # 2. False Positive
    ax1 = axes[1]
    if cases["false_positives"]:
        fp = cases["false_positives"][0]
        img = fp["image"].permute(1, 2, 0).numpy().clip(0, 1)
        ax1.imshow(img)
        pb = fp["pred_box"]
        rect = patches.Rectangle(
            (pb[0], pb[1]), pb[2] - pb[0], pb[3] - pb[1],
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax1.add_patch(rect)
        cname = cats[fp["pred_label"]] if fp["pred_label"] < len(cats) else f"cls_{fp['pred_label']}"
        ax1.set_title(f"False Positive (Spurious Alarm)\nPred: {cname} (Conf={fp['score']:.2f})", color="darkred", weight="bold")
    else:
        ax1.text(0.5, 0.5, "No False Positives extracted", ha="center")
    ax1.axis("off")

    # 3. Classification / Localization Error
    ax2 = axes[2]
    if cases["classification_errors"]:
        ce = cases["classification_errors"][0]
        img = ce["image"].permute(1, 2, 0).numpy().clip(0, 1)
        ax2.imshow(img)
        gb = ce["gt_box"]
        pb = ce["pred_box"]
        rect_gt = patches.Rectangle(
            (gb[0], gb[1]), gb[2] - gb[0], gb[3] - gb[1],
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        rect_pr = patches.Rectangle(
            (pb[0], pb[1]), pb[2] - pb[0], pb[3] - pb[1],
            linewidth=2, edgecolor="red", linestyle="--", facecolor="none",
        )
        ax2.add_patch(rect_gt)
        ax2.add_patch(rect_pr)
        gt_name = cats[ce["gt_label"]] if ce["gt_label"] < len(cats) else f"cls_{ce['gt_label']}"
        pr_name = cats[ce["pred_label"]] if ce["pred_label"] < len(cats) else f"cls_{ce['pred_label']}"
        ax2.set_title(f"Inter-Class Confusion\nGT: {gt_name} (Green) vs. Pred: {pr_name} (Red, IoU={ce['iou']:.2f})", color="purple", weight="bold")
    else:
        ax2.text(0.5, 0.5, "No Confusion Cases extracted", ha="center")
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

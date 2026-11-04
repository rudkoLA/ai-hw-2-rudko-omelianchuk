"""AI Lab 2 package."""

from src.setup_utils import setup_environment, ensure_dataset
from src.dataset import VisDroneDataset, build_dataloader, get_hardware_config
from src.preprocessing import letterbox, letterbox_image
from src.evaluation import (
    evaluate_coco_metrics,
    format_coco_predictions,
    unletterbox_bboxes,
    generate_yolo_submission,
    validate_submission_file,
)
from src.models import (
    build_faster_rcnn_v2,
    train_one_epoch,
    evaluate_model_performance,
    analyze_failure_cases
)

__all__ = [
    "setup_environment",
    "ensure_dataset",
    "VisDroneDataset",
    "build_dataloader",
    "get_hardware_config",
    "letterbox",
    "letterbox_image",
    "evaluate_coco_metrics",
    "format_coco_predictions",
    "unletterbox_bboxes",
    "generate_yolo_submission",
    "validate_submission_file",
    "build_faster_rcnn_v2",
    "train_one_epoch",
    "evaluate_model_performance",
    "analyze_failure_cases",
]

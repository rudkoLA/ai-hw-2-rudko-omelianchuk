"""AI Lab 2 package."""

from src.dataset import (
    VisDroneDataset,
    build_dataloader,
    create_visdrone_loaders,
    get_hardware_config,
    select_frames,
)
from src.eda import compute_sequence_heterogeneity, run_full_eda
from src.evaluation import (
    evaluate_coco_metrics,
    format_coco_predictions,
    generate_faster_rcnn_submission,
    generate_retinanet_submission,
    generate_yolo_submission,
    unletterbox_bboxes,
    validate_submission_file,
)
from src.models import (
    analyze_failure_cases,
    build_faster_rcnn_v2,
    build_retinanet_v2,
    evaluate_detector_model,
    evaluate_model_performance,
    plot_failure_cases,
    train_detector_model,
    train_one_epoch,
)
from src.preprocessing import (
    RandomZoomCrop,
    get_dfine_train_transforms,
    get_train_transforms,
    get_val_transforms,
    letterbox,
    letterbox_image,
)
from src.setup_utils import (
    COMPETITION_NAME,
    ensure_dataset,
    resolve_storage,
    setup_environment,
    setup_kaggle_credentials,
    use_native_batchnorm,
)

__all__ = [
    "COMPETITION_NAME",
    "RandomZoomCrop",
    "VisDroneDataset",
    "analyze_failure_cases",
    "build_dataloader",
    "build_faster_rcnn_v2",
    "build_retinanet_v2",
    "compute_sequence_heterogeneity",
    "create_visdrone_loaders",
    "ensure_dataset",
    "evaluate_coco_metrics",
    "evaluate_detector_model",
    "evaluate_model_performance",
    "format_coco_predictions",
    "generate_faster_rcnn_submission",
    "generate_retinanet_submission",
    "generate_yolo_submission",
    "get_dfine_train_transforms",
    "get_hardware_config",
    "get_train_transforms",
    "get_val_transforms",
    "letterbox",
    "letterbox_image",
    "plot_failure_cases",
    "resolve_storage",
    "run_full_eda",
    "select_frames",
    "setup_environment",
    "setup_kaggle_credentials",
    "train_detector_model",
    "train_one_epoch",
    "unletterbox_bboxes",
    "use_native_batchnorm",
    "validate_submission_file",
]

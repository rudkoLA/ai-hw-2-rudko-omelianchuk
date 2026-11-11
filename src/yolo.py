"""Ultralytics YOLO (YOLO26 / YOLOv8-style heads) with tunable label assignment and box IoU loss.

Ultralytics hard-codes these inside its loss, so they are injected through a model subclass
(criterion) and a trainer subclass (builds that model). The classes live in a module so pickled
checkpoints can be re-loaded.
"""

from typing import Optional, Type

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import BboxLoss, E2ELoss
from ultralytics.utils.metrics import bbox_iou

from src.ignore_mask import masked_train_dataset

IOU_TYPES = ("CIoU", "DIoU", "GIoU", "IoU")


class IoUTypeBboxLoss(BboxLoss):
    """BboxLoss with a selectable IoU variant for the box term (Ultralytics always uses CIoU).

    The distance term (DFL, or L1 when reg_max == 1 as in YOLO26) is unchanged.
    """

    def __init__(self, reg_max: int = 16, iou_type: str = "CIoU") -> None:
        super().__init__(reg_max)
        assert iou_type in IOU_TYPES, f"iou_type must be one of {IOU_TYPES}"
        self.iou_type = iou_type

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dist = super().forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
            target_scores_sum, fg_mask, imgsz, stride,
        )
        if self.iou_type == "CIoU":
            return loss_iou, loss_dist
        fg = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg].sum(-1, keepdim=True)
        flags = {} if self.iou_type == "IoU" else {self.iou_type: True}
        iou = bbox_iou(pred_bboxes[fg], target_bboxes[fg], xywh=False, **flags)
        return ((1.0 - iou) * weight).sum() / target_scores_sum, loss_dist


class TunedDetectionModel(DetectionModel):
    """DetectionModel whose criterion uses configurable label assignment and box IoU loss.

    Args:
        tal_topk: Positives per object for the one-to-many branch (Ultralytics default 10).
        stal_min_side: Small-target-aware assignment: GT sides below this many pixels are enlarged to it
            when selecting candidate grid points (None = Ultralytics default, the 2nd-smallest stride;
            0 disables it).
        iou_type: Box IoU loss variant ("CIoU", "DIoU", "GIoU", "IoU").
    """

    def __init__(self, cfg="yolo26s.yaml", ch=3, nc=None, verbose=True,
                 tal_topk: int = 10, stal_min_side: Optional[float] = None, iou_type: str = "CIoU"):
        self.tal_topk = tal_topk
        self.stal_min_side = stal_min_side
        self.iou_type = iou_type
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        criterion = super().init_criterion()
        # YOLO26 (end2end) has a one-to-many and a one-to-one branch; YOLOv8-style heads only one.
        one2many = criterion.one2many if isinstance(criterion, E2ELoss) else criterion
        branches = [criterion.one2many, criterion.one2one] if isinstance(criterion, E2ELoss) else [criterion]

        one2many.assigner.topk = one2many.assigner.topk2 = self.tal_topk
        for branch in branches:
            if self.stal_min_side is not None:
                branch.assigner.stride_val = self.stal_min_side
            branch.bbox_loss = IoUTypeBboxLoss(branch.reg_max, self.iou_type).to(branch.device)
        return criterion


def make_yolo_trainer(
    tal_topk: int = 10,
    stal_min_side: Optional[float] = None,
    iou_type: str = "CIoU",
    mask_ignore: bool = False,
) -> Type[DetectionTrainer]:
    """Builds a DetectionTrainer subclass that trains a TunedDetectionModel with the given settings.

    mask_ignore: Fill ignored regions of training images with gray (see src/ignore_mask.py).
    """

    class Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = TunedDetectionModel(
                cfg,
                ch=self.data["channels"],
                nc=self.data["nc"],
                verbose=verbose and RANK == -1,
                tal_topk=tal_topk,
                stal_min_side=stal_min_side,
                iou_type=iou_type,
            )
            model = self.set_model_names_for_load(model)
            if weights:
                model.load(weights)
            return model

        def build_dataset(self, img_path, mode="train", batch=None):
            with masked_train_dataset(mask_ignore, mode):
                return super().build_dataset(img_path, mode, batch)

    return Trainer

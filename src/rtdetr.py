"""RT-DETR (Ultralytics) with tunable query count, denoising, matching costs, loss gains and backbone LR.

Ultralytics exposes none of these as train() arguments, so they are injected through
a model subclass (criterion) and a trainer subclass (model config + optimizer groups).
The classes live in a module (not a notebook) so pickled checkpoints can be re-loaded.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Type

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.nn.tasks import RTDETRDetectionModel, yaml_model_load
from ultralytics.utils import RANK

# Ultralytics defaults (DETRLoss / HungarianMatcher)
DEFAULT_MATCHER_COST = {"class": 2.0, "bbox": 5.0, "giou": 2.0}
DEFAULT_LOSS_GAIN = {"class": 1.0, "bbox": 5.0, "giou": 2.0}


def build_rtdetr_cfg(
    base_cfg: str = "rtdetr-l.yaml",
    num_queries: int = 300,
    num_denoising: int = 100,
    hidden_dim: int = 256,
) -> Dict[str, Any]:
    """Returns an RT-DETR model config dict with a custom number of queries / denoising queries.

    RTDETRDecoder positional args: nc, ch (inserted by parse_model), hd, nq, ndp, nh, ndl,
    d_ffn, dropout, act, eval_idx, nd. Only hd / nq / nd are changed; the rest are defaults.
    Pretrained weights still load: initial queries are selected from encoder tokens
    (learnt_init_query=False), so no parameter depends on the number of queries.
    """
    cfg = deepcopy(yaml_model_load(base_cfg))
    head = cfg["head"][-1]
    assert head[2] == "RTDETRDecoder", f"Unexpected last head layer: {head}"
    head[3] = ["nc", hidden_dim, num_queries, 4, 8, 6, 1024, 0.0, None, -1, num_denoising]
    return cfg


class TunedRTDETRModel(RTDETRDetectionModel):
    """RTDETRDetectionModel whose loss uses configurable Hungarian matching costs and loss gains."""

    def __init__(
        self,
        cfg: Any = "rtdetr-l.yaml",
        ch: int = 3,
        nc: Optional[int] = None,
        verbose: bool = True,
        matcher_cost: Optional[Dict[str, float]] = None,
        loss_gain: Optional[Dict[str, float]] = None,
    ) -> None:
        self.matcher_cost = dict(matcher_cost or DEFAULT_MATCHER_COST)
        self.loss_gain = dict(loss_gain or DEFAULT_LOSS_GAIN)
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self) -> RTDETRDetectionLoss:
        criterion = RTDETRDetectionLoss(nc=self.nc, loss_gain=self.loss_gain, use_vfl=True)
        criterion.matcher = HungarianMatcher(cost_gain=self.matcher_cost)
        return criterion


def make_rtdetr_trainer(
    model_cfg: Dict[str, Any],
    matcher_cost: Optional[Dict[str, float]] = None,
    loss_gain: Optional[Dict[str, float]] = None,
    backbone_lr_mult: float = 1.0,
) -> Type[RTDETRTrainer]:
    """Builds an RTDETRTrainer subclass bound to the given model config and tuning knobs.

    Args:
        model_cfg: Config dict from build_rtdetr_cfg (defines num_queries / num_denoising).
        matcher_cost: Hungarian matching cost weights {"class", "bbox", "giou"}.
        loss_gain: Loss weights {"class", "bbox", "giou"}.
        backbone_lr_mult: Backbone LR = lr0 * backbone_lr_mult (rest of the model uses lr0).
    """
    n_backbone = len(model_cfg["backbone"])

    class Trainer(RTDETRTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # Always build from model_cfg: Model.train() passes the pretrained model's own yaml.
            model = TunedRTDETRModel(
                deepcopy(model_cfg),
                ch=self.data["channels"],
                nc=self.data["nc"],
                verbose=verbose and RANK == -1,
                matcher_cost=matcher_cost,
                loss_gain=loss_gain,
            )
            model = self.set_model_names_for_load(model)
            if weights:
                model.load(weights)
            return model

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
            if backbone_lr_mult == 1.0:
                return optimizer
            # Split every Ultralytics group (weight / bn / bias) into backbone and non-backbone parts.
            backbone_ids = {id(p) for p in model.model[:n_backbone].parameters()}
            old_groups, optimizer.param_groups = optimizer.param_groups, []
            for g in old_groups:
                opts = {k: v for k, v in g.items() if k != "params"}
                rest = [p for p in g["params"] if id(p) not in backbone_ids]
                bb = [p for p in g["params"] if id(p) in backbone_ids]
                if rest:
                    optimizer.add_param_group({**opts, "params": rest})
                if bb:
                    optimizer.add_param_group({**opts, "params": bb, "lr": g["lr"] * backbone_lr_mult})
            return optimizer

    return Trainer

from functools import cached_property

from torch import Tensor, nn
from torchmetrics import (
    AUROC,
    Accuracy,
    Dice,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)

from lymph_nodes.model import LymphNodesModel
from lymph_nodes.modeling.losses import DiceFocalLoss
from lymph_nodes.typing import Outputs


class SegmentationModel(LymphNodesModel):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__(backbone)

    @cached_property
    def criterion(self) -> nn.Module:
        return DiceFocalLoss()

    @cached_property
    def val_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "DICE": Dice(num_classes=2),
                "IOU": JaccardIndex(task="binary"),
                "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary"),
                "precision": Precision("binary"),
                "recall": Recall("binary"),
            }
        )

    def forward(self, x: Tensor) -> Outputs:
        return self.backbone(x)

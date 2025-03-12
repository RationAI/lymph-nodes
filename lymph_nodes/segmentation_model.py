from functools import cached_property

from torch import Tensor, nn
from torchmetrics import (
    # AUROC,
    Accuracy,
    # Dice,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)

from lymph_nodes.model import LymphNodesModel
from lymph_nodes.modeling.losses import DiceFocalLoss
from lymph_nodes.typing import Outputs


class SegmentationModel(LymphNodesModel):
    @cached_property
    def criterion(self) -> nn.Module:
        return DiceFocalLoss()

    def get_val_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                # "DICE": Dice(),
                "IOU": JaccardIndex(task="binary"),
                # "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary"),
                "precision": Precision("binary"),
                "recall": Recall("binary"),
            }
        )

    def forward(self, x: Tensor) -> Outputs:
        return self.backbone(x).squeeze(1)

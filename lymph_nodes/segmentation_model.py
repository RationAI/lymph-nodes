from torch import Tensor, nn
from torchmetrics import (
    # AUROC,
    Accuracy,
    Dice,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)

from lymph_nodes.model import LymphNodesModel
from lymph_nodes.modeling import SetCriterion
from lymph_nodes.typing import Outputs


class SegmentationModel(LymphNodesModel):
    def __init__(
        self, model: nn.Module, criterion: SetCriterion, warmup_epochs: int
    ) -> None:
        super().__init__(model, criterion, warmup_epochs)
        self.criterion = criterion

    def get_val_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "DICE": Dice(),
                "IOU": JaccardIndex(task="binary"),
                # "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary"),
                "precision": Precision("binary"),
                "recall": Recall("binary"),
            }
        )

    def forward(self, x: Tensor) -> Outputs:
        return self.model(x).squeeze(1)

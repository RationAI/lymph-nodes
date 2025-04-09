from torch import Tensor, nn
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

from lymph_nodes.model import LymphNodesModel
from lymph_nodes.modeling import SetCriterion
from lymph_nodes.typing import Outputs


class ClassificationModel(LymphNodesModel):
    def __init__(self, model: nn.Module, warmup_epochs: int) -> None:
        super().__init__(
            model,
            criterion=SetCriterion(
                {"bce": 1},
                {"bce": nn.BCELoss()},
            ),
            warmup_epochs=warmup_epochs,
        )

    def get_val_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary"),
                "precision": Precision("binary"),
                "recall": Recall("binary"),
            }
        )

    def forward(self, x: Tensor) -> Outputs:
        return self.model(x)

from functools import cached_property

from torch import Tensor, nn
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

from lymph_nodes.model import LymphNodesModel
from lymph_nodes.modeling.binary_classifier import BinaryClassifier
from lymph_nodes.typing import Outputs


class ClassificationModel(LymphNodesModel):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__(backbone)
        self.decode_head = BinaryClassifier()

    @cached_property
    def criterion(self) -> nn.Module:
        return nn.BCELoss()

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
        features = self.backbone(x)
        return self.decode_head(features)

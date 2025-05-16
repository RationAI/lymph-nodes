from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCohenKappa,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from lymph_nodes.model import LymphNodesModel
from lymph_nodes.modeling import CriterionLoss, SetCriterion
from lymph_nodes.typing import ClsSample, Metadata, Outputs, Targets


class ClassificationModel(LymphNodesModel):
    def __init__(self, model: nn.Module, warmup_epochs: int) -> None:
        super().__init__(
            model,
            criterion=SetCriterion(
                losses=[
                    CriterionLoss(
                        weight=1, name="bce", loss=nn.BCELoss(), kind="labels"
                    )
                ]
            ),
            warmup_epochs=warmup_epochs,
        )

    def get_val_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "AUC": BinaryAUROC(),
                "F1": BinaryF1Score(),
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "specificity": BinarySpecificity(),
                "CohensKappa": BinaryCohenKappa(),
            }
        )

    def read_batch(self, batch: ClsSample) -> tuple[Tensor, Targets, Metadata]:
        inputs, labels, metadata = batch
        return inputs, Targets(labels=labels), metadata

    def update_metrics(
        self,
        metrics: MetricCollection | LazyMetricDict,
        outputs: Outputs,
        targets: Targets,
        key: str | None = None,
    ) -> None:
        if isinstance(metrics, LazyMetricDict):
            metrics.update(outputs.labels, targets.labels, key=key)
        else:
            metrics.update(outputs.labels, targets.labels)

    def forward(self, x: Tensor) -> Outputs:
        return Outputs(labels=self.model(x))

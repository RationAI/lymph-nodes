import torch
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryCohenKappa,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.segmentation import MeanIoU

# from torchmetrics.segmentation import DiceScore
from lymph_nodes.model import LymphNodesModel
from lymph_nodes.modeling import SetCriterion
from lymph_nodes.typing import Metadata, Outputs, SegSample, Targets


class SegmentationModel(LymphNodesModel):
    def __init__(
        self, model: nn.Module, criterion: SetCriterion, warmup_epochs: int
    ) -> None:
        super().__init__(model, criterion, warmup_epochs)
        self.criterion = criterion

    def get_val_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "nIOU": MeanIoU(),
                "IOU": BinaryJaccardIndex(),
                "F1": BinaryF1Score(),
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "specificity": BinarySpecificity(),
                "CohensKappa": BinaryCohenKappa(),
            }
        )

    def read_batch(self, batch: SegSample) -> tuple[Tensor, Targets, Metadata]:
        inputs, masks, labels, metadata = batch
        return inputs, Targets(labels=labels, masks=masks), metadata

    def update_metrics(
        self,
        metrics: MetricCollection | LazyMetricDict,
        outputs: Outputs,
        targets: Targets,
        key: str | None = None,
    ) -> None:
        if isinstance(metrics, LazyMetricDict):
            metrics.update(outputs.masks, targets.masks.to(torch.uint8), key=key)
        else:
            metrics.update(outputs.masks, targets.masks.to(torch.uint8))

    def forward(self, x: Tensor) -> Outputs:
        return self.model(x)

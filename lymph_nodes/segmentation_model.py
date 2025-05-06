import torch
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torchmetrics import (
    # AUROC,
    Accuracy,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)

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
                # "DICE": DiceScore(num_classes=1),
                "IOU": JaccardIndex(task="binary"),
                # "AUC": AUROC("binary"),
                "accuracy": Accuracy("binary"),
                "precision": Precision("binary"),
                "recall": Recall("binary"),
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

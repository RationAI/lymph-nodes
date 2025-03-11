from abc import ABC, abstractmethod
from functools import cached_property

from lightning import LightningModule
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
import torch

from lymph_nodes.typing import Input


class LymphNodesModel(LightningModule, ABC):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

        self.val_metrics = self.get_val_metrics()
        self.test_metrics = LazyMetricDict(self.val_metrics.clone())
        self.test_metrics_collection = self.val_metrics.clone()
        self.val_metrics.prefix = "validation/"

    @cached_property
    @abstractmethod
    def criterion(self) -> nn.Module: ...

    @abstractmethod
    def get_val_metrics(self) -> MetricCollection: ...

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log(
            "train/loss", loss, batch_size=len(inputs), on_step=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch: Input) -> None:
        inputs, targets, _ = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)

        self.log(
            "validation/loss",
            loss,
            batch_size=len(inputs),
            on_epoch=True,
            prog_bar=True,
        )

        self.val_metrics.update(outputs, targets.to(torch.uint8))
        self.log_dict(self.val_metrics, batch_size=len(inputs), on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=0.0001)

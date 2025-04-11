from abc import ABC, abstractmethod
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from rationai.mlkit.metrics import LazyMetricDict
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import Tensor, nn
from torchmetrics import MetricCollection

from lymph_nodes.modeling import SetCriterion
from lymph_nodes.typing import Input, PredictInput


class LymphNodesModel(LightningModule, ABC):
    def __init__(
        self, model: nn.Module, criterion: SetCriterion, warmup_epochs: int
    ) -> None:
        super().__init__()

        # It has to be named backbone because of FineTuner
        self.backbone = model.backbone

        self.model = model
        self.criterion = criterion
        self.warmup_epochs = warmup_epochs

        self.val_metrics = self.get_val_metrics()
        self.test_metrics = LazyMetricDict(self.val_metrics.clone())
        self.test_metrics_collection = self.val_metrics.clone()
        self.val_metrics.prefix = "validation/"

    @abstractmethod
    def get_val_metrics(self) -> MetricCollection: ...

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)

        losses = self.criterion(outputs, targets)

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            batch_size=len(inputs),
            on_step=True,
            prog_bar=True,
        )

        return losses["loss"]

    def validation_step(self, batch: Input) -> None:
        inputs, targets, _ = batch
        outputs = self(inputs)

        losses = self.criterion(outputs, targets)

        self.log_dict(
            {f"validation/{k}": v for k, v in losses.items()},
            batch_size=len(inputs),
            on_epoch=True,
        )

        self.val_metrics.update(outputs, targets.to(torch.uint8))
        self.log_dict(self.val_metrics, batch_size=len(inputs), on_epoch=True)

    def test_step(
        self, batch: Input, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        inputs, targets, metadata = batch
        outputs = self(inputs)

        for output, target, slide in zip(
            outputs, targets, metadata["slide"], strict=False
        ):
            self.test_metrics.update(output, target.to(torch.uint8), key=slide)

        self.test_metrics_collection.update(outputs, targets.to(torch.uint8))
        self.log_dict(
            self.test_metrics_collection, batch_size=len(inputs), on_epoch=True
        )

        return outputs

    def predict_step(
        self, batch: PredictInput, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        inputs, metadata = batch
        return self(inputs)

    def on_test_epoch_end(self) -> None:
        for key, metrics in self.test_metrics.compute().items():
            table = {k: v.item() for k, v in metrics.items()}
            self.logger.log_table({"slide": key, **table}, "test_metrics.json")
        self.test_metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4,
            weight_decay=1e-4,
        )

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.trainer.max_epochs,
            lr_min=1e-6,
            warmup_lr_init=1e-7,
            warmup_t=self.warmup_epochs,
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(
        self, scheduler: CosineLRScheduler, metric: Any | None
    ) -> None:
        scheduler.step(epoch=self.current_epoch)

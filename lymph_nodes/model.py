from abc import ABC, abstractmethod
from functools import cached_property

import torch
from lightning import LightningModule
from rationai.mlkit.metrics import LazyMetricDict
from torch import Tensor, nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from lymph_nodes.typing import Input


class LymphNodesModel(LightningModule, ABC):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.model = backbone

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
        inputs, targets, metadata = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)

        if loss > 3:
            print(
                inputs,
                targets,
                outputs,
                metadata,
                targets.view(targets.shape[0], -1).sum(dim=1),
            )

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

    # def predict_step(
    #     self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    # ) -> Outputs:
    #     inputs, metadata = batch
    #     outputs = self(inputs)

    #     for output, slide_id, x, y in zip(
    #         outputs, metadata["slide_id"], metadata["x"], metadata["y"], strict=False
    #     ):
    #         self.inference_data.append(
    #             Prediction(
    #                 slide_id=slide_id,
    #                 x=x.item(),
    #                 y=y.item(),
    #                 probability=output.item(),
    #             )
    #         )

    #     return outputs

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.95))

    def on_test_epoch_end(self) -> None:
        for key, metrics in self.test_metrics.compute().items():
            table = {k: v.item() for k, v in metrics.items()}
            self.logger.log_table({"slide": key, **table}, "test_metrics.json")
        self.test_metrics.reset()

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

    from lymph_nodes.typing import Batch, Input, Outputs


class MetaArch(LightningModule):
    """Binary (cancer / healthy) tile classifier on top of a precomputed embedding.

    ``backbone`` is typically ``nn.Identity()`` here: the embedding (GigaPath/UNI2/
    Virchow2) was already computed by the preprocessing pipeline, not by this model.
    ``decode_head`` is the actual trainable classifier mapping embedding -> 1 logit.
    """

    def __init__(self, backbone: nn.Module, decode_head: nn.Module, lr: float = 1e-3) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.lr = lr

        # Class imbalance (~20:1 healthy:cancer) is handled at the data level by
        # StratifiedEpochSampler (lymph_nodes/data/samplers/stratified_sampler.py),
        # not here — avoid double-compensating with e.g. BCEWithLogitsLoss(pos_weight=...).
        self.criterion = nn.BCEWithLogitsLoss()

        # torchmetrics' Binary* metrics auto-apply sigmoid to inputs outside [0, 1],
        # so raw logits can be passed directly without a separate activation step.
        metrics = MetricCollection({
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "auroc": BinaryAUROC(),
        })
        self.val_metrics = metrics.clone(prefix="validation/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: Input) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features).squeeze(-1)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        inputs, targets, _meta = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets.float())
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        inputs, targets, _meta = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets.float())
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        self.val_metrics.update(outputs, targets)
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        inputs, targets, _meta = batch
        outputs = self(inputs)

        self.test_metrics.update(outputs, targets)
        self.log_dict(self.test_metrics, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_f1_score,
    binary_precision,
    binary_recall,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from torch import Tensor
    from torch.optim import Optimizer

    from lymph_nodes.typedefs import Batch, Input, Outputs


# The overall metrics again, for each cohort of the evaluated data (see MetaArch).
_COHORT_METRICS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "accuracy": binary_accuracy,
    "precision": binary_precision,
    "recall": binary_recall,
    "f1": binary_f1_score,
    "auroc": binary_auroc,
}


class MetaArch(LightningModule):
    """Binary (cancer / healthy) tile classifier on top of a precomputed embedding.

    ``backbone`` is typically ``nn.Identity()`` here: the embedding (GigaPath/UNI2/
    Virchow2) was already computed by the preprocessing pipeline, not by this model.
    ``decode_head`` is the actual trainable classifier mapping embedding -> 1 logit.

    The final evaluations (``validate`` and ``test``, not the validation during ``fit``)
    also log every metric per cohort of the evaluated data (the slides' cohort_id), as
    ``<phase>/<cohort>/<metric>`` next to the overall ``<phase>/<metric>``: an
    evaluation set can mix cohorts that behave very differently. The validation and test
    steps return their logits for callbacks, e.g. lymph_nodes.callbacks.PredictionMasks.
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

        # cohort -> (logits, targets) of the running final evaluation
        self._cohort_outputs: dict[str, tuple[list[Tensor], list[Tensor]]] = {}

    def forward(self, x: Input) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features).squeeze(-1)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        inputs, targets, _meta = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets.float())
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Outputs:
        inputs, targets, meta = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets.float())
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        self.val_metrics.update(outputs, targets)
        self.log_dict(self.val_metrics, on_epoch=True)

        if self.trainer.state.fn == TrainerFn.VALIDATING:
            self._collect_by_cohort(outputs, targets, meta)
        return outputs

    def on_validation_epoch_end(self) -> None:
        if self.trainer.state.fn == TrainerFn.VALIDATING:
            self._log_by_cohort("validation")

    def test_step(self, batch: Batch, batch_idx: int) -> Outputs:
        inputs, targets, meta = batch
        outputs = self(inputs)

        self.test_metrics.update(outputs, targets)
        self.log_dict(self.test_metrics, on_epoch=True)

        self._collect_by_cohort(outputs, targets, meta)
        return outputs

    def on_test_epoch_end(self) -> None:
        self._log_by_cohort("test")

    def _collect_by_cohort(self, outputs: Tensor, targets: Tensor, meta: dict[str, Any]) -> None:
        cohorts: list[str] = meta["cohort"]
        for cohort in set(cohorts):
            mask = torch.tensor([c == cohort for c in cohorts], device=outputs.device)
            logits, labels = self._cohort_outputs.setdefault(cohort, ([], []))
            logits.append(outputs[mask].detach().float().cpu())
            labels.append(targets[mask].detach().cpu())

    def _log_by_cohort(self, phase: str) -> None:
        for cohort, (logits, labels) in sorted(self._cohort_outputs.items()):
            preds, target = torch.cat(logits), torch.cat(labels).int()
            values: dict[str, float | Tensor] = {
                f"{phase}/{cohort}/n_tiles": float(len(target)),
                f"{phase}/{cohort}/n_positive": float(target.sum()),
            }
            for name, metric in _COHORT_METRICS.items():
                # AUROC is undefined for a cohort holding a single class.
                if name != "auroc" or target.unique().numel() == 2:
                    values[f"{phase}/{cohort}/{name}"] = metric(preds, target)
            self.log_dict(values)
        self._cohort_outputs.clear()

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

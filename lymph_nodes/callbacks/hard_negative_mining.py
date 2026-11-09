from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

    from lymph_nodes.data.samplers.hard_negative_bank import HardNegativeBank


class HardNegativeMining(Callback):
    """Feeds the training sampler's HardNegativeBank with the model's own scores.

    After every training step: each negative tile of the batch (meta ``label`` at most
    0.5 — not cancer, no cells pasted) is scored by its most confident cell, from the
    logits the step returns (SegmentationMetaArch: ``{"loss", "logits"}``), and the
    scores go to the bank of the training sampler (StratifiedEpochSampler's
    ``hard_negative_bank``), with the tiles' dataset indices (meta ``index``) and vivid
    flags (meta ``vivid``). No extra forward pass: the scores are the step's own. At
    each epoch's end the bank's state is logged as ``hard_negatives/<stat>``.
    """

    def _bank(self, trainer: Trainer) -> HardNegativeBank | None:
        sampler = getattr(trainer.datamodule, "train_sampler", None)  # type: ignore[attr-defined]
        return getattr(sampler, "hard_negative_bank", None)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        bank = self._bank(trainer)
        if bank is None or not isinstance(outputs, dict) or "logits" not in outputs:
            return
        meta = batch[2]
        logits = outputs["logits"].detach().float()
        scores = torch.sigmoid(logits).flatten(1).amax(dim=1).cpu().numpy()
        negative = torch.as_tensor(meta["label"]).float().cpu().numpy() <= 0.5
        vivid = (
            torch.as_tensor(meta["vivid"]).cpu().numpy().astype(bool)
            if "vivid" in meta
            else np.zeros(len(scores), dtype=bool)
        )
        indices = torch.as_tensor(meta["index"]).cpu().numpy()
        bank.update(indices[negative], scores[negative], vivid[negative])

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        bank = self._bank(trainer)
        if bank is not None:
            pl_module.log_dict(
                {f"hard_negatives/{key}": value for key, value in bank.stats().items()}
            )

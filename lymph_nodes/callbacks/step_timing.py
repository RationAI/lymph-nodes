from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback


if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class StepTiming(Callback):
    """Where a training batch's time goes, per epoch, as ``timing/<phase>_ms`` (means).

    Phases between the training hooks, the GPU synchronized at each (``synchronize``)
    so that its queued work counts where it was issued:

    - ``wait``: the previous batch's end to this one's start — the data loader, and
      the other callbacks' and the module's ``on_train_batch_end``;
    - ``forward``: the training step up to the loss (forward, loss, logging);
    - ``backward``: the backward pass;
    - ``step``: the backward's end to the batch's end — the optimizer step and
      zero_grad (every ``accumulate_grad_batches`` batches), the LR scheduler.

    Added 2026-11-24 for the LoRA fits, whose batches slowed 4.6-fold over five
    epochs with the GPU and the loader idle (HANDOFF.md, step 5f). Costs a few
    synchronizations per batch.
    """

    def __init__(self, synchronize: bool = True) -> None:
        self.synchronize = synchronize
        self._last_end: float | None = None
        self._marks: dict[str, float] = {}
        self._sums: dict[str, float] = {}
        self._count = 0

    def _now(self) -> float:
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _add(self, phase: str, seconds: float) -> None:
        self._sums[phase] = self._sums.get(phase, 0.0) + seconds

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._last_end, self._sums, self._count = None, {}, 0

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        now = self._now()
        if self._last_end is not None:
            self._add("wait", now - self._last_end)
        self._marks = {"start": now}

    def on_before_backward(
        self, trainer: Trainer, pl_module: LightningModule, loss: torch.Tensor
    ) -> None:
        self._marks["backward"] = self._now()

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._marks["step"] = self._now()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        now = self._now()
        marks = self._marks
        if {"start", "backward", "step"} <= marks.keys():
            self._add("forward", marks["backward"] - marks["start"])
            self._add("backward", marks["step"] - marks["backward"])
            self._add("step", now - marks["step"])
            self._count += 1
        self._last_end = now

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._count:
            pl_module.log_dict(
                {
                    f"timing/{phase}_ms": 1000 * total / self._count
                    for phase, total in self._sums.items()
                },
                on_step=False,
                on_epoch=True,
            )

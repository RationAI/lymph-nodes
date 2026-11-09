from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn.functional as F
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from lymph_nodes.meta_arch import MetaArch, _kept, _select


if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch import Tensor

    from lymph_nodes.typedefs import Batch, Input, Outputs


def tile_scores(
    probabilities: Tensor,
    meta: dict[str, Any] | None = None,
    mode: Literal["max", "mean"] = "max",
    region: Literal["full", "centre"] = "centre",
) -> Tensor:
    """Tile scores (B,) from per-patch probabilities (B, g, g).

    ``mode``: the most confident patch (``max``) or the predicted coverage (``mean``).
    ``region``: all patches (``full``), or, where tiles overlap, each tile's centre
    (``centre``): meta ``centre_clip`` (TileSegmentationDataset), the patches cut off
    each side — then every point of the slide is scored by the one tile in which it
    has the most context, not by up to four, from their edges too. The same for the
    tile, slide and lesion metrics (SegmentationMetaArch, SlideMetrics, LesionMetrics).
    """
    flat = probabilities.flatten(1)
    clip = meta.get("centre_clip") if meta is not None and region == "centre" else None
    if clip is None or not bool(torch.as_tensor(clip).any()):
        return flat.amax(dim=1) if mode == "max" else flat.mean(dim=1)
    g = probabilities.shape[-1]
    cut = torch.as_tensor(clip, device=flat.device).view(-1, 1)
    index = torch.arange(g, device=flat.device)
    inside = (index >= cut) & (index < g - cut)  # (B, g)
    centre = (inside[:, :, None] & inside[:, None, :]).flatten(1)
    if mode == "max":
        return torch.where(centre, flat, -torch.inf).amax(dim=1)
    return (flat * centre).sum(dim=1) / centre.sum(dim=1)


class SegmentationMetaArch(MetaArch):
    """Per-patch (segmentation) cancer classifier on the patch tokens of a frozen model.

    ``backbone`` maps tiles to a (B, C, g, g) feature map — FrozenFoundationModel, its
    patch tokens — and ``decode_head`` that to one logit per patch, (B, 1, g, g): e.g.
    a 1x1 convolution, a linear probe of every patch token; a head with ``uses_image``
    gets the tiles as well (UpsamplingHead). Trained per patch against
    the target grid (TileSegmentationDataset: each patch's coverage, a soft target)
    with ``criterion``: BCE when not given, e.g. DiceFocalLoss
    (configs/model/foundation_segmentation.yaml). A model whose patch grid differs from
    the targets' (GigaPath: 16 px patches, 14 x 14 per 224 px tile, against the
    16 x 16 grids of 14 px) has its logits bilinearly resized to the target grid first.

    Evaluated per patch (``<phase>/patch/{auroc,f1}``, a patch counting as cancer when
    its target is above 0.5) and per tile, like MetaArch — overall and per cohort, the
    same tile labels (meta["label"]) and PredictionMasks — with a tile scored by its
    patches' probabilities: ``tile_score="max"``, its most confident patch (detection:
    is there cancer in the tile?), or ``"mean"``, its predicted coverage. The validation
    and test steps return the per-patch logits (B, g, g), so the masks show patches.
    Samples flagged ``ignore`` are left out of the loss and the metrics, as in MetaArch.
    Patches whose label is unknown (meta ``ignore_grid``, TileSegmentationDataset's
    ``ignore_grids``) are weighted by the share known, 1 - ignore, in the loss (which
    must take ``weights``, as DiceFocalLoss) and left out of the patch metrics when
    mostly unknown. A criterion with ``components`` (DiceFocalLoss) has its terms logged
    apart too, ``<phase>/loss/<term>``; Dice only for batches with tumour (on a
    tumour-free one it measures the probabilities' sum, not an overlap).
    Samples with a second target grid, the pathologists' outlines when the targets are
    the refined labels (meta ``outline_grid``, TileSegmentationDataset's
    ``outline_grids``) get the patch metrics against it too, ``<phase>/patch/outline/``;
    and the patches of tiles not flagged ambiguous (meta ``ambiguous``, TileLabels) get
    them again as ``<phase>/patch/unambiguous/``. ``tile_region``: whether overlapping
    tiles are scored by their centre (see tile_scores).

    Training: AdamW at ``lr``, constant or (``lr_schedule="cosine"``) after
    ``warmup_steps`` of linear warm-up down a cosine to ``min_lr_ratio`` x ``lr`` at the
    last step (max_epochs). With ``ema_decay``, an exponential moving average of the
    head's weights (``ema_head``) is kept after every step and evaluates (validation,
    test, checkpoints' use): it smooths the epoch-to-epoch swings of the head itself.

    The frozen backbone is left out of checkpoints (it is reloaded from the hub).
    """

    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        tile_score: Literal["max", "mean"] = "max",
        criterion: nn.Module | None = None,
        tile_region: Literal["full", "centre"] = "centre",
        lr_schedule: Literal["constant", "cosine"] = "constant",
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        ema_decay: float | None = None,
    ) -> None:
        super().__init__(
            backbone,
            decode_head,
            lr=lr,
            weight_decay=weight_decay,
            criterion=criterion,
        )
        self.tile_score = tile_score
        self.tile_region = tile_region
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.ema_head = (
            AveragedModel(decode_head, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
            if ema_decay
            else None
        )
        # Checkpoints hold the head only (on_save_checkpoint).
        self.strict_loading = False

        # Binned AUROC: exact AUROC would keep every patch of the epoch (256 per tile).
        patch_metrics = MetricCollection(
            {"auroc": BinaryAUROC(thresholds=200), "f1": BinaryF1Score()}
        )
        self.val_patch_metrics = _patch_metric_sets(patch_metrics, "validation")
        self.test_patch_metrics = _patch_metric_sets(patch_metrics, "test")

    def forward(self, x: Input) -> Outputs:
        features = self.backbone(x)
        # Training steps the head; its weight average, if kept, evaluates.
        head = (
            self.ema_head.module
            if self.ema_head is not None and not self.training
            else self.decode_head
        )
        if getattr(head, "uses_image", False):  # e.g. UpsamplingHead
            return head(features, x).squeeze(1)
        return head(features).squeeze(1)

    def configure_optimizers(self) -> OptimizerLRScheduler:  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.decode_head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_schedule == "constant":
            return optimizer
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup, floor = self.warmup_steps, self.min_lr_ratio

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min((step - warmup) / max(total - warmup, 1), 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.ema_head is not None:
            self.ema_head.update_parameters(self.decode_head)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint["state_dict"]
        for key in [key for key in state if key.startswith("backbone.")]:
            del state[key]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # A checkpoint without a weight average (trained without one) evaluates with
        # its head's own weights.
        state = checkpoint["state_dict"]
        if self.ema_head is None or any(k.startswith("ema_head.") for k in state):
            return
        for key in [k for k in state if k.startswith("decode_head.")]:
            state["ema_head.module." + key.removeprefix("decode_head.")] = state[key]
        state["ema_head.n_averaged"] = torch.tensor(1, dtype=torch.long)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        tiles, grids, meta = batch
        logits = self._logits(tiles, grids)
        ignore = meta.get("ignore_grid")
        loss = self._loss(logits, grids, ignore)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self._log_terms("train", logits.detach(), grids, ignore, on_step=True)
        return loss

    def _log_terms(
        self,
        phase: str,
        logits: Tensor,
        grids: Tensor,
        ignore: Tensor | None,
        **log_kwargs: Any,
    ) -> None:
        """The criterion's terms apart, if it has ``components`` (see the class)."""
        components = getattr(self.criterion, "components", None)
        if components is None:
            return
        weights = None if ignore is None else 1 - ignore.to(grids.device).float()
        with torch.no_grad():
            terms = components(logits, grids.float(), weights)
            known = grids.float() if weights is None else grids.float() * weights
            has_tumour = bool(known.sum() > 0)
        for name, value in terms.items():
            if name != "dice" or has_tumour:
                self.log(f"{phase}/loss/{name}", value, **log_kwargs)

    def _loss(self, logits: Tensor, grids: Tensor, ignore: Tensor | None) -> Tensor:
        """The criterion, the patches weighted by the share of them with a known label."""
        if ignore is None:
            return self.criterion(logits, grids.float())
        weights = 1 - ignore.to(grids.device).float()
        return self.criterion(logits, grids.float(), weights=weights)

    def validation_step(self, batch: Batch, batch_idx: int) -> Outputs:
        logits, keep = self._evaluate(batch, "validation", self.val_patch_metrics)
        if keep is not None:
            scores = self._tile_scores(logits[keep], _select(batch[2], keep))
            labels = batch[2]["label"][keep] > 0.5
            self.val_metrics.update(scores, labels)
            self.log_dict(self.val_metrics, on_epoch=True)
            if self.trainer.state.fn == TrainerFn.VALIDATING:
                self._collect_by_cohort(scores, labels, _select(batch[2], keep))
        return logits

    def test_step(self, batch: Batch, batch_idx: int) -> Outputs:
        logits, keep = self._evaluate(batch, "test", self.test_patch_metrics)
        if keep is not None:
            scores = self._tile_scores(logits[keep], _select(batch[2], keep))
            labels = batch[2]["label"][keep] > 0.5
            self.test_metrics.update(scores, labels)
            self.log_dict(self.test_metrics, on_epoch=True)
            self._collect_by_cohort(scores, labels, _select(batch[2], keep))
        return logits

    def _logits(self, tiles: Tensor, grids: Tensor) -> Tensor:
        """Per-patch logits on the grid of the targets (B, g, g)."""
        logits = self(tiles)
        if logits.shape[-2:] != grids.shape[-2:]:
            logits = F.interpolate(
                logits.unsqueeze(1), size=grids.shape[-2:], mode="bilinear"
            ).squeeze(1)
        return logits

    def _evaluate(
        self,
        batch: Batch,
        phase: str,
        patch_metrics: nn.ModuleDict,
    ) -> tuple[Tensor, Tensor | None]:
        """Per-patch logits of all tiles and the mask of the counted ones; logs the patch side."""
        tiles, grids, meta = batch
        logits = self._logits(tiles, grids)
        keep = _kept(meta, logits)
        if keep is None:
            return logits, None
        ignore = meta.get("ignore_grid")
        if ignore is not None:
            ignore = ignore.to(grids.device)[keep.to(grids.device)]
        self.log(
            f"{phase}/loss",
            self._loss(logits[keep], grids[keep], ignore),
            on_epoch=True,
            prog_bar=True,
            batch_size=int(keep.sum()),
        )
        self._log_terms(
            phase,
            logits[keep],
            grids[keep],
            ignore,
            on_epoch=True,
            batch_size=int(keep.sum()),
        )

        probabilities = torch.sigmoid(logits[keep].float())
        targets = grids[keep] > 0.5
        sets = [("all", probabilities, targets)]
        if "outline_grid" in meta:
            outline = meta["outline_grid"][keep.to(meta["outline_grid"].device)]
            sets.append(("outline", probabilities, outline.to(targets.device) > 0.5))
        clear = ~torch.as_tensor(meta["ambiguous"], dtype=torch.bool)[keep.cpu()]
        if clear.any():
            clear = clear.to(targets.device)
            sets.append(("unambiguous", probabilities[clear], targets[clear]))
        # Patches mostly of unknown label count in no patch metric.
        known = None if ignore is None else ignore < 0.5
        for name, preds, target in sets:
            metrics = cast("MetricCollection", patch_metrics[name])
            if known is None:
                metrics.update(preds.flatten(), target.flatten())
            else:
                mask = known[clear] if name == "unambiguous" else known
                if mask.any():
                    metrics.update(preds[mask], target[mask])
            self.log_dict(metrics, on_epoch=True)
        return logits, keep

    def _tile_scores(self, logits: Tensor, meta: dict[str, Any]) -> Tensor:
        """Tile probabilities from per-patch logits (B, g, g) (see tile_scores)."""
        return tile_scores(
            torch.sigmoid(logits.float()), meta, self.tile_score, self.tile_region
        )


def _patch_metric_sets(metrics: MetricCollection, phase: str) -> nn.ModuleDict:
    """The patch metrics of a phase: against the targets, the outlines, and unambiguous."""
    return nn.ModuleDict(
        {
            "all": metrics.clone(prefix=f"{phase}/patch/"),
            "outline": metrics.clone(prefix=f"{phase}/patch/outline/"),
            "unambiguous": metrics.clone(prefix=f"{phase}/patch/unambiguous/"),
        }
    )

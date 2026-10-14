from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from lightning.pytorch.trainer.states import TrainerFn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from lymph_nodes.meta_arch import MetaArch


if TYPE_CHECKING:
    from torch import Tensor, nn

    from lymph_nodes.typedefs import Batch, Input, Outputs


class SegmentationMetaArch(MetaArch):
    """Per-patch (segmentation) cancer classifier on the patch tokens of a frozen model.

    ``backbone`` maps tiles to a (B, C, g, g) feature map — FrozenFoundationModel, its
    patch tokens — and ``decode_head`` that to one logit per patch, (B, 1, g, g): e.g.
    a 1x1 convolution, a linear probe of every patch token. Trained per patch against
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
    ) -> None:
        super().__init__(
            backbone,
            decode_head,
            lr=lr,
            weight_decay=weight_decay,
            criterion=criterion,
        )
        self.tile_score = tile_score
        # Checkpoints hold the head only (on_save_checkpoint).
        self.strict_loading = False

        # Binned AUROC: exact AUROC would keep every patch of the epoch (256 per tile).
        patch_metrics = MetricCollection(
            {"auroc": BinaryAUROC(thresholds=200), "f1": BinaryF1Score()}
        )
        self.val_patch_metrics = patch_metrics.clone(prefix="validation/patch/")
        self.test_patch_metrics = patch_metrics.clone(prefix="test/patch/")

    def forward(self, x: Input) -> Outputs:
        return self.decode_head(self.backbone(x)).squeeze(1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.decode_head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint["state_dict"]
        for key in [key for key in state if key.startswith("backbone.")]:
            del state[key]

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        tiles, grids, _meta = batch
        loss = self.criterion(self._logits(tiles, grids), grids.float())
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Outputs:
        logits, scores, labels = self._evaluate(
            batch, "validation", self.val_patch_metrics
        )
        self.val_metrics.update(scores, labels)
        self.log_dict(self.val_metrics, on_epoch=True)
        if self.trainer.state.fn == TrainerFn.VALIDATING:
            self._collect_by_cohort(scores, labels, batch[2])
        return logits

    def test_step(self, batch: Batch, batch_idx: int) -> Outputs:
        logits, scores, labels = self._evaluate(batch, "test", self.test_patch_metrics)
        self.test_metrics.update(scores, labels)
        self.log_dict(self.test_metrics, on_epoch=True)
        self._collect_by_cohort(scores, labels, batch[2])
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
        self, batch: Batch, phase: str, patch_metrics: MetricCollection
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-patch logits, tile scores (probabilities) and tile labels; logs the patch side."""
        tiles, grids, meta = batch
        logits = self._logits(tiles, grids)
        self.log(
            f"{phase}/loss",
            self.criterion(logits, grids.float()),
            on_epoch=True,
            prog_bar=True,
        )

        probabilities = torch.sigmoid(logits.float())
        patch_metrics.update(probabilities.flatten(), grids.flatten() > 0.5)
        self.log_dict(patch_metrics, on_epoch=True)

        flat = probabilities.flatten(1)
        scores = flat.amax(dim=1) if self.tile_score == "max" else flat.mean(dim=1)
        return logits, scores, meta["label"] > 0.5

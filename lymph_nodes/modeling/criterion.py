from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn


if TYPE_CHECKING:
    from torch import Tensor


class DiceFocalLoss(nn.Module):
    """Weighted sum of a focal loss and a soft Dice loss, on logits and soft targets.

    Made for SegmentationMetaArch: per-patch logits (B, g, g) against coverage targets
    in [0, 1] (TileSegmentationDataset), but any shape works. Both terms take soft
    targets as they are:

    - focal: binary cross-entropy weighted by ``|target - p| ** gamma``, the soft-target
      form of the focal loss (Li et al., 2020, "Generalized Focal Loss"): for a hard
      target it is the usual ``(1 - p_t) ** gamma`` (Lin et al., 2017), for a soft one
      it vanishes where the prediction matches the coverage. It down-weights the easy
      patches, which are most of them: lymph node tissue and stroma far from any
      epithelium. ``alpha``, if given, weights the positive part of each target
      (``1 - alpha`` the negative part), like BCEWithLogitsLoss's ``pos_weight``; the
      sampler already balances the classes (StratifiedEpochSampler), so it is off by
      default.
    - Dice: ``1 - (2 * sum(p * t) + smooth) / (sum(p) + sum(t) + smooth)`` over the
      whole batch. Over the batch rather than per tile: most tiles (every lymph node
      training tile) have no positive patch, where a per-tile Dice is undefined or,
      smoothed, a second and stronger push towards zero than the focal term already
      gives. Dice measures overlap, so the rare positive patches count as much as the
      plentiful negative ones.

    - tile (``tile_weight``, off by default): per tile, the mean of its ``tile_topk``
      highest logits against the tile's target, its highest cell coverage (binary
      cross-entropy). It trains what the evaluation scores — a tile by its most
      confident cells: a negative tile's few highest cells pushed down (the false
      positive regions), a positive one's up — without Dice's pull towards zero over
      every cell of tumour-free batches, nor its weighting by area (a sheet of TMA
      tumour outweighs a cell or two). The mean of a few rather than the maximum keeps
      the gradient steady; ``tile_topk`` ~ the cells of an isolated tumour cell
      (8: two to three cells of 3.4 um at grid 32).

    ``focal_weight``, ``dice_weight`` and ``tile_weight`` scale the terms; a weight of 0
    drops its term (``components`` gives them apart, for logging). ``weights``, if given (the shape of the targets, in [0, 1]), weights each
    patch in both terms, e.g. 0 for a patch whose label is unknown (SegmentationMetaArch:
    1 - its ignore grid); the focal term is then their weighted mean. Computed in
    float32 whatever the training precision.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
        alpha: float | None = None,
        smooth: float = 1.0,
        tile_weight: float = 0.0,
        tile_topk: int = 8,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tile_weight = tile_weight
        self.tile_topk = tile_topk
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(
        self, logits: Tensor, targets: Tensor, weights: Tensor | None = None
    ) -> Tensor:
        parts = self.components(logits, targets, weights)
        loss = logits.new_zeros((), dtype=torch.float32)
        for name, weight in self._weights().items():
            if name in parts:
                loss = loss + weight * parts[name]
        return loss

    def _weights(self) -> dict[str, float]:
        return {
            "focal": self.focal_weight,
            "dice": self.dice_weight,
            "tile": self.tile_weight,
        }

    def components(
        self, logits: Tensor, targets: Tensor, weights: Tensor | None = None
    ) -> dict[str, Tensor]:
        """The terms with a weight, unweighted: {"focal", "dice", "tile"}."""
        logits, targets = logits.float(), targets.float()
        weights = torch.ones_like(targets) if weights is None else weights.float()
        terms = {"focal": self._focal, "dice": self._dice, "tile": self._tile}
        return {
            name: terms[name](logits, targets, weights)
            for name, weight in self._weights().items()
            if weight
        }

    def _focal(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (targets - torch.sigmoid(logits)).abs().pow(self.gamma) * bce
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def _tile(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        flat = logits.flatten(1)
        known = weights.flatten(1) >= 0.5
        k = min(self.tile_topk, flat.shape[1])
        # The k highest of the cells with a known label; tiles with fewer left out.
        top = flat.masked_fill(~known, -1e4).topk(k, dim=1).values.mean(dim=1)
        target = (targets * weights).flatten(1).amax(dim=1).clamp(0, 1)
        valid = (known.sum(dim=1) >= k).float()
        loss = F.binary_cross_entropy_with_logits(top, target, reduction="none")
        return (loss * valid).sum() / valid.sum().clamp_min(1)

    def _dice(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        probabilities = torch.sigmoid(logits) * weights
        targets = targets * weights
        overlap = 2 * (probabilities * targets).sum() + self.smooth
        return 1 - overlap / (probabilities.sum() + targets.sum() + self.smooth)

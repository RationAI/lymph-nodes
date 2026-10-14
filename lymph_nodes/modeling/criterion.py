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

    ``focal_weight`` and ``dice_weight`` scale the two terms; a weight of 0 drops its
    term. Computed in float32 whatever the training precision.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
        alpha: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits, targets = logits.float(), targets.float()
        loss = logits.new_zeros(())
        if self.focal_weight:
            loss = loss + self.focal_weight * self._focal(logits, targets)
        if self.dice_weight:
            loss = loss + self.dice_weight * self._dice(logits, targets)
        return loss

    def _focal(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (targets - torch.sigmoid(logits)).abs().pow(self.gamma) * bce
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        return loss.mean()

    def _dice(self, logits: Tensor, targets: Tensor) -> Tensor:
        probabilities = torch.sigmoid(logits)
        overlap = 2 * (probabilities * targets).sum() + self.smooth
        return 1 - overlap / (probabilities.sum() + targets.sum() + self.smooth)

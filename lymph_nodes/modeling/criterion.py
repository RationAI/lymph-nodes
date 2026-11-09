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
    - asl (``asl_weight``, off by default): the asymmetric loss (Ridnik et al., 2021), an
      alternative to the focal term — focusing ``asl_gamma_pos`` on the positive part
      of each target and ``asl_gamma_neg`` on the negative part, the latter on a
      probability shifted down by ``asl_clip``: a negative patch below it costs
      nothing (the easy majority drops out entirely), and the gradient of a negative
      scored very high, likely a mislabelled one (an isolated tumour cell missed on a
      "negative" slide), shrinks rather than grows.
    - rank (``rank_weight``, off by default): the tiles ranked as the evaluation reads
      them — tumour above the most suspicious benign tissue, sensitivity at a few false
      positives (a partial AUC). Each tile's score is its tile term's (the mean of its
      ``tile_topk`` highest logits); every positive tile of the batch (highest target
      at least ``rank_positive``) is paired with the batch's ``rank_negatives``
      highest-scoring negative tiles (highest target at most ``rank_negative``) and
      each pair costs ``softplus(rank_margin - (positive - negative))``: nothing once
      the tumour leads by the margin, most where benign tissue outranks it.

    ``focal_weight``, ``dice_weight``, ``tile_weight``, ``asl_weight`` and
    ``rank_weight`` scale the terms; a weight of 0 drops its term (``components`` gives
    them apart, for logging). ``weights``, if given (the shape of the targets, in
    [0, 1]), weights each patch in every term, e.g. 0 for a patch whose label is
    unknown (SegmentationMetaArch: 1 - its ignore grid); the per-patch terms are then
    their weighted means. Computed in float32 whatever the training precision.
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
        asl_weight: float = 0.0,
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        rank_weight: float = 0.0,
        rank_margin: float = 1.0,
        rank_negatives: int = 32,
        rank_positive: float = 0.5,
        rank_negative: float = 0.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tile_weight = tile_weight
        self.tile_topk = tile_topk
        self.asl_weight = asl_weight
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_clip = asl_clip
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin
        self.rank_negatives = rank_negatives
        self.rank_positive = rank_positive
        self.rank_negative = rank_negative
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
            "asl": self.asl_weight,
            "rank": self.rank_weight,
        }

    def components(
        self, logits: Tensor, targets: Tensor, weights: Tensor | None = None
    ) -> dict[str, Tensor]:
        """The terms with a weight, unweighted: {"focal", "dice", "tile", "asl", "rank"}."""
        logits, targets = logits.float(), targets.float()
        weights = torch.ones_like(targets) if weights is None else weights.float()
        terms = {
            "focal": self._focal,
            "dice": self._dice,
            "tile": self._tile,
            "asl": self._asl,
            "rank": self._rank,
        }
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

    def _tile_scores(
        self, logits: Tensor, targets: Tensor, weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per tile: score (top-k mean of known logits), target, enough known cells.

        The score is the mean of its ``tile_topk`` highest logits among the cells with
        a known label, the target its highest known coverage.
        """
        flat = logits.flatten(1)
        known = weights.flatten(1) >= 0.5
        k = min(self.tile_topk, flat.shape[1])
        top = flat.masked_fill(~known, -1e4).topk(k, dim=1).values.mean(dim=1)
        target = (targets * weights).flatten(1).amax(dim=1).clamp(0, 1)
        return top, target, known.sum(dim=1) >= k

    def _tile(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        top, target, valid = self._tile_scores(logits, targets, weights)
        loss = F.binary_cross_entropy_with_logits(top, target, reduction="none")
        return (loss * valid.float()).sum() / valid.float().sum().clamp_min(1)

    def _asl(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        p = torch.sigmoid(logits)
        positive = (1 - p).pow(self.asl_gamma_pos) * F.logsigmoid(logits)
        shifted = (p - self.asl_clip).clamp_min(0)
        negative = shifted.pow(self.asl_gamma_neg) * torch.log(
            (1 - shifted).clamp_min(1e-8)
        )
        loss = -(targets * positive + (1 - targets) * negative)
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def _rank(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        score, target, valid = self._tile_scores(logits, targets, weights)
        positive = score[valid & (target >= self.rank_positive)]
        negative = score[valid & (target <= self.rank_negative)]
        if len(positive) == 0 or len(negative) == 0:
            return logits.new_zeros(())
        hardest = negative.topk(min(self.rank_negatives, len(negative))).values
        gap = positive[:, None] - hardest[None, :]
        return F.softplus(self.rank_margin - gap).mean()

    def _dice(self, logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        probabilities = torch.sigmoid(logits) * weights
        targets = targets * weights
        overlap = 2 * (probabilities * targets).sum() + self.smooth
        return 1 - overlap / (probabilities.sum() + targets.sum() + self.smooth)

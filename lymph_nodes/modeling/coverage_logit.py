from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn


if TYPE_CHECKING:
    from torch import Tensor


class CoverageLogit(nn.Module):
    """A coverage fraction (B, 1) as a logit, for rule-based baselines in MetaArch.

    ``log(coverage / threshold)``: the probability ``coverage / (coverage + threshold)``
    is 0.5 exactly at ``threshold``, so the thresholded metrics mean "coverage above
    the threshold"; the ranking (AUROC) is the coverage's own. No parameters — used
    with ``mode=validate`` only, e.g. the brown-staining baseline (``brownish_coverage``
    as the input, configs/experiment/training/brown_baseline.yaml).
    """

    def __init__(self, threshold: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, coverage: Tensor) -> Tensor:
        return torch.log(coverage.float().clamp_min(self.eps) / self.threshold)

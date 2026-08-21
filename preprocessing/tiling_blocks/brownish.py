from __future__ import annotations

from typing import Any

import numpy as np
from ray.data import Dataset

from preprocessing.tiling_blocks.tiling_block import TilingBlock


class BrownishCoverage(TilingBlock):
    """Fraction of tile pixels that are DAB-positive, computed via Ruifrok-Johnston OD projection.

    Requires a pixel column loaded before this step (e.g. via ReadTiles).
    """

    _DAB_OD = np.array([0.268, 0.570, 0.776], dtype=np.float32)
    _DAB_OD /= np.linalg.norm(_DAB_OD)

    def __init__(
        self,
        name: str,
        image_col: str,
        dab_threshold: float = 0.15,
    ) -> None:
        self._name = name
        self._image_col = image_col
        self._dab_threshold = dab_threshold

    def apply(self, tiles: Dataset) -> Dataset:
        def _compute(row: dict[str, Any]) -> dict[str, Any]:
            od = -np.log10(np.clip(row[self._image_col].astype(np.float32) / 255.0, 1e-6, 1.0))
            row[self._name] = float(((od * self._DAB_OD).sum(axis=-1) > self._dab_threshold).mean())
            return row

        return tiles.map(_compute)

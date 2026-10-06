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

    # Optical density of every possible uint8 intensity, so per-pixel log10 becomes a lookup.
    _OD_LUT = -np.log10(np.clip(np.arange(256, dtype=np.float32) / 255.0, 1e-6, 1.0))

    def __init__(
        self,
        name: str,
        image_col: str,
        dab_threshold: float = 0.15,
        batch_size: int = 256,
    ) -> None:
        self._name = name
        self._image_col = image_col
        self._dab_threshold = dab_threshold
        self._batch_size = batch_size

    def apply(self, tiles: Dataset) -> Dataset:
        image_col, name, threshold = self._image_col, self._name, self._dab_threshold
        od_lut, dab_od = self._OD_LUT, self._DAB_OD

        def _compute(batch: dict[str, Any]) -> dict[str, Any]:
            images = np.stack(batch[image_col])  # (B, H, W, 3) uint8
            dab = od_lut[images] @ dab_od  # (B, H, W)
            batch[name] = (dab > threshold).mean(axis=(1, 2))
            return batch

        return tiles.map_batches(_compute, batch_format="numpy", batch_size=self._batch_size)

from __future__ import annotations

from typing import Any

import numpy as np
from ray.data import Dataset

from preprocessing.tiling_blocks.tiling_block import TilingBlock


class BrownishCoverage(TilingBlock):
    """Fraction of tile pixels that look brown (DAB-stained), by fixed RGB thresholds.

    The same per-pixel rule as the ``color_separation_masks`` behind the mask-based
    ``brownish_coverage`` (minus their resolution and the morphology they leave
    commented out): a pixel is brown when it is

    - warm overall: ``R + G > B + warmth``,
    - red-dominant, with blue not far above green: ``R > G >= B - blue_tolerance``,
    - mid-dark: ``red_min <= R < red_max``.

    Blue haematoxylin nuclei fail ``R > G`` and the warmth test, white background and
    very light staining fail ``R < red_max``, near-black debris fails ``R >= red_min``.
    Unlike projecting optical density onto the DAB stain vector, this doesn't count
    dark haematoxylin as brown — but it is conservative: faint DAB (R >= red_max) is
    missed. Defaults are the thresholds the colour-separation masks were made with.

    Requires a pixel column loaded before this step (e.g. via ReadTiles).
    """

    def __init__(
        self,
        name: str,
        image_col: str,
        warmth: int = 50,
        blue_tolerance: int = 10,
        red_min: int = 50,
        red_max: int = 175,
        batch_size: int = 256,
    ) -> None:
        self._name = name
        self._image_col = image_col
        self._warmth = warmth
        self._blue_tolerance = blue_tolerance
        self._red_min = red_min
        self._red_max = red_max
        self._batch_size = batch_size

    def apply(self, tiles: Dataset) -> Dataset:
        image_col, name = self._image_col, self._name
        warmth, blue_tolerance = self._warmth, self._blue_tolerance
        red_min, red_max = self._red_min, self._red_max

        def _compute(batch: dict[str, Any]) -> dict[str, Any]:
            # int16: R + G and B - tolerance must not wrap around as uint8 would.
            images = np.stack(batch[image_col]).astype(np.int16)  # (B, H, W, 3)
            red, green, blue = images[..., 0], images[..., 1], images[..., 2]
            brown = (
                (red + green > blue + warmth)
                & (red > green)
                & (green >= blue - blue_tolerance)
                & (red >= red_min)
                & (red < red_max)
            )
            batch[name] = brown.mean(axis=(1, 2))
            return batch

        return tiles.map_batches(_compute, batch_format="numpy", batch_size=self._batch_size)

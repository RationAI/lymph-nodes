from __future__ import annotations

from typing import Any

import numpy as np
from ray.data import Dataset

from preprocessing.tiling_blocks.tiling_block import TilingBlock
from preprocessing.white_balance import WHITE_POINT_COLUMNS, apply_white_point


class WhiteBalance(TilingBlock):
    """Corrects the pixels of ``image_col`` by the slide's white point.

    Underexposed scans (e.g. FN Brno) are scaled so that their glass becomes white
    (preprocessing.white_balance), before colour-based blocks such as BrownishCoverage
    see them. The white point comes with each tile from the slides table (``white_r``,
    ``white_g``, ``white_b``); tiles without one are left unchanged. Place it right after
    ReadTiles.
    """

    def __init__(self, image_col: str, batch_size: int = 256) -> None:
        self._image_col = image_col
        self._batch_size = batch_size

    def apply(self, tiles: Dataset) -> Dataset:
        image_col = self._image_col

        def _correct(batch: dict[str, Any]) -> dict[str, Any]:
            if not all(column in batch for column in WHITE_POINT_COLUMNS):
                return batch
            whites = np.stack(
                [
                    np.asarray(batch[column], dtype=np.float32)
                    for column in WHITE_POINT_COLUMNS
                ],
                axis=1,
            )
            images = list(batch[image_col])
            for i, white in enumerate(whites):
                if np.isfinite(white).all():
                    images[i] = apply_white_point(np.asarray(images[i]), white)
            batch[image_col] = np.stack(images)
            return batch

        return tiles.map_batches(
            _correct, batch_format="numpy", batch_size=self._batch_size
        )

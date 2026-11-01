from __future__ import annotations

from typing import Any

import numpy as np
from ray.data import Dataset

from preprocessing.staining import stained_cells
from preprocessing.tiling_blocks.tiling_block import TilingBlock


class DabCoverage(TilingBlock):
    """Fraction of the tile in cytokeratin-stained cells, by colour deconvolution.

    preprocessing.staining.stained_cells on each tile at its own resolution (the
    ``mpp_x`` column): DAB density (skimage's rgb2hed) above ``threshold``, extended by
    hysteresis to ``low_threshold``, specks under ``min_area_um2`` dropped, nuclei filled,
    grown by ``grow_um``. The defaults are those of the refined annotation masks
    (configs/preprocessing/refined_annotation_masks.yaml, ``stain``), so the coverage
    counts the same cells as the refined labels: faint DAB too, dark haematoxylin not.
    A single stained cell covers ~0.002-0.007 of a 224 px tile at 0.5 um/px.

    Requires a pixel column loaded before this step (e.g. via ReadTiles).
    """

    def __init__(
        self,
        name: str,
        image_col: str,
        threshold: float = 0.03,
        low_threshold: float = 0.022,
        min_area_um2: float = 10.0,
        hole_area_um2: float = 200.0,
        grow_um: float = 0.0,
        batch_size: int = 256,
    ) -> None:
        self._name = name
        self._image_col = image_col
        self._stain = {
            "threshold": threshold,
            "low_threshold": low_threshold,
            "min_area_um2": min_area_um2,
            "hole_area_um2": hole_area_um2,
            "grow_um": grow_um,
        }
        self._batch_size = batch_size

    def apply(self, tiles: Dataset) -> Dataset:
        image_col, name, stain = self._image_col, self._name, self._stain

        def _compute(batch: dict[str, Any]) -> dict[str, Any]:
            batch[name] = np.array(
                [
                    stained_cells(
                        np.asarray(image)[..., :3], float(mpp), **stain
                    ).mean()
                    for image, mpp in zip(batch[image_col], batch["mpp_x"], strict=True)
                ],
                dtype=np.float32,
            )
            return batch

        return tiles.map_batches(
            _compute, batch_format="numpy", batch_size=self._batch_size
        )

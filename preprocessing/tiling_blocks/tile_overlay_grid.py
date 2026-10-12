from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from omegaconf import DictConfig
from ray.data import Dataset
from shapely import Polygon

from preprocessing.tiling_blocks.tile_overlay import OverlayCoverage


def grid_coverage(inside: np.ndarray, valid: np.ndarray, grid: int) -> np.ndarray:
    """Per-cell coverage of an (H, W) tile split into grid x grid cells.

    ``inside`` flags the pixels counted, ``valid`` the pixels within the ROI; a cell's
    coverage is its counted fraction of valid pixels (0 if it has none). H and W need
    not be multiples of ``grid`` — e.g. a mask at a slightly different mpp than the
    slide level — the cell edges are then rounded.
    """
    rows = np.linspace(0, inside.shape[0], grid + 1).round().astype(int)[:-1]
    cols = np.linspace(0, inside.shape[1], grid + 1).round().astype(int)[:-1]

    def cell_sums(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(
            np.add.reduceat(values.astype(np.int64), rows, axis=0), cols, axis=1
        )

    counted, total = cell_sums(inside & valid), cell_sums(valid)
    return np.divide(counted, total, out=np.zeros((grid, grid)), where=total > 0)


class OverlayGrid(OverlayCoverage):
    """Like OverlayCoverage, but the coverage of each cell of a grid x grid split of the tile.

    For dense (segmentation) targets: with ``grid`` = tile extent / patch size of the
    foundation model (224 / 14 = 16), a cell is exactly one of its patch tokens. The
    column holds ``grid * grid`` uint8 values (coverage * 255) in row-major order, top
    row first; null for tiles whose slide has no mask (dropped if ``mandatory``).
    Labels are counted as in OverlayCoverage (``min_label``/``max_label``).
    """

    def __init__(
        self,
        name: str,
        uri: str | None,
        roi: Polygon | DictConfig,
        grid: int = 16,
        suffix: str = "tiff",
        mandatory: bool = False,
        min_label: int | None = None,
        max_label: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            uri=uri,
            roi=roi,
            suffix=suffix,
            mandatory=mandatory,
            min_label=min_label,
            max_label=max_label,
        )
        self._grid = grid

    def apply(self, tiles: Dataset) -> Dataset:
        # Private, but the undecorated reader behind ratiopath's tile_overlay UDF: the
        # overlay of each tile as a masked (H, W, C) array (mask = outside the ROI).
        from ratiopath.tiling.overlays import _tile_overlay

        overlay_paths = self._overlay_paths
        roi = self._roi
        name = self._name
        grid = self._grid
        min_label = -np.inf if self._min_label is None else self._min_label
        max_label = np.inf if self._max_label is None else self._max_label
        mandatory = self._mandatory
        column_type = pa.list_(pa.uint8(), grid * grid)

        def add_grid(batch: pa.Table) -> pa.Table:
            stems = [Path(p).stem for p in batch["path"].to_pylist()]
            mask_exists = pa.array(
                [stem in overlay_paths for stem in stems], type=pa.bool_()
            )

            with_mask = batch.filter(mask_exists)
            without_mask = batch.filter(pc.invert(mask_exists))

            values: list[list[int]] = []
            if with_mask.num_rows:
                resolved_paths = pa.array(
                    [
                        str(overlay_paths[stem])
                        for stem in stems
                        if stem in overlay_paths
                    ],
                    type=pa.string(),
                )
                overlays = _tile_overlay(
                    roi,
                    resolved_paths,
                    with_mask["tile_x"],
                    with_mask["tile_y"],
                    with_mask["mpp_x"],
                    with_mask["mpp_y"],
                )
                for overlay in overlays:
                    labels = overlay.data[..., 0]
                    inside = (labels >= min_label) & (labels <= max_label)
                    valid = ~np.ma.getmaskarray(overlay)[..., 0]
                    coverage = grid_coverage(inside, valid, grid)
                    values.append(
                        np.round(coverage * 255).astype(np.uint8).ravel().tolist()
                    )

            with_mask = with_mask.append_column(
                name, pa.array(values, type=column_type)
            )
            if mandatory:
                return with_mask

            without_mask = without_mask.append_column(
                name, pa.nulls(without_mask.num_rows, type=column_type)
            )
            return pa.concat_tables(
                [with_mask, without_mask.select(with_mask.column_names)]
            )

        return tiles.map_batches(add_grid, batch_format="pyarrow")

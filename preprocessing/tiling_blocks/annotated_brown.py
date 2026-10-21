from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from omegaconf import DictConfig
from ray.data import Dataset
from shapely import Polygon

from preprocessing.tiling_blocks.brownish import brown_pixels
from preprocessing.tiling_blocks.tile_overlay import OverlayCoverage
from preprocessing.tiling_blocks.tile_overlay_grid import grid_coverage


class AnnotatedBrown(OverlayCoverage):
    """The annotation narrowed to stained tissue: refined evaluation labels.

    The pathologists' outlines are rough: drawn generously around a focus, they take in
    the lymphoid tissue around and between the stained cells (on the IHC slides ~17% of
    the tiles on an outline's edge have no brown pixel at all). This keeps the pixels of
    a tile that lie inside the annotation (labels counted as in OverlayCoverage) *and*
    within ``radius_um`` of one of the tile's own brown pixels (BrownishCoverage's rule;
    a square of that half-side) — a cell's reach, so a stained cell counts whole, its
    nucleus included, while unstained tissue the outline swept in does not. Faint DAB
    the rule misses drops out with it.

    Writes ``<name>_coverage``, the kept fraction of the tile's ROI pixels (as
    OverlayCoverage), and ``<name>_grid``, the same per cell of a grid x grid split (as
    OverlayGrid, uint8 coverage * 255). Tiles of slides without an annotation mask get
    NaN coverage and an all-zero grid rather than nulls: a grid mixing values and nulls
    can't leave numpy batches. With ``annotation_column`` (the annotation's
    OverlayCoverage), only the tiles it touches are read; the rest of an annotated
    slide's tiles get 0.

    Needs the tile's pixels (``image_col``, e.g. ReadTiles), ``path`` and the tiling
    level's ``mpp_x``/``mpp_y``; place it before the blocks that drop them.
    """

    def __init__(
        self,
        name: str,
        uri: str | None,
        roi: Polygon | DictConfig,
        image_col: str,
        grid: int = 16,
        radius_um: float = 10.0,
        annotation_column: str | None = None,
        suffix: str = "tiff",
        min_label: int | None = None,
        max_label: int | None = None,
        batch_size: int = 256,
    ) -> None:
        super().__init__(
            name=name,
            uri=uri,
            roi=roi,
            suffix=suffix,
            mandatory=False,
            min_label=min_label,
            max_label=max_label,
        )
        self._image_col = image_col
        self._grid = grid
        self._radius_um = radius_um
        self._annotation_column = annotation_column
        self._batch_size = batch_size

    def apply(self, tiles: Dataset) -> Dataset:
        # The undecorated reader behind ratiopath's tile_overlay UDF, as in OverlayGrid.
        from ratiopath.tiling.overlays import _tile_overlay

        overlay_paths, roi, name = self._overlay_paths, self._roi, self._name
        image_col, grid, radius_um = self._image_col, self._grid, self._radius_um
        annotation_column = self._annotation_column
        min_label = -np.inf if self._min_label is None else self._min_label
        max_label = np.inf if self._max_label is None else self._max_label

        def compute(batch: dict[str, Any]) -> dict[str, Any]:
            n = len(batch["path"])
            stems = [Path(str(path)).stem for path in batch["path"]]
            annotated = np.array([stem in overlay_paths for stem in stems], dtype=bool)
            coverage = np.where(annotated, 0.0, np.nan)
            grids = np.zeros((n, grid * grid), dtype=np.uint8)

            read = annotated.copy()
            if annotation_column is not None:
                touched = np.asarray(batch[annotation_column], dtype=np.float64)
                read &= np.nan_to_num(touched) > 0
            rows = np.flatnonzero(read)
            if len(rows):
                overlays = _tile_overlay(
                    roi,
                    pa.array(
                        [str(overlay_paths[stems[i]]) for i in rows], type=pa.string()
                    ),
                    pa.array(np.asarray(batch["tile_x"])[rows]),
                    pa.array(np.asarray(batch["tile_y"])[rows]),
                    pa.array(np.asarray(batch["mpp_x"])[rows]),
                    pa.array(np.asarray(batch["mpp_y"])[rows]),
                )
                for i, overlay in zip(rows, overlays, strict=True):
                    image = np.asarray(batch[image_col][i])
                    labels = overlay.data[..., 0]
                    valid = ~np.ma.getmaskarray(overlay)[..., 0]
                    inside = (labels >= min_label) & (labels <= max_label) & valid
                    shape = image.shape[:2]
                    radius = round(radius_um / float(batch["mpp_x"][i]))
                    near_brown = dilate(brown_pixels(image[..., :3]), radius)
                    kept = resize_nearest(inside, shape) & near_brown
                    valid_px = resize_nearest(valid, shape)
                    coverage[i] = kept.sum() / max(int(valid_px.sum()), 1)
                    cells = grid_coverage(kept, valid_px, grid)
                    grids[i] = np.round(cells * 255).astype(np.uint8).ravel()

            batch[f"{name}_coverage"] = coverage
            batch[f"{name}_grid"] = grids
            return batch

        return tiles.map_batches(
            compute, batch_format="numpy", batch_size=self._batch_size
        )


def resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``mask`` (h, w) sampled onto ``shape`` (H, W) by nearest neighbour."""
    rows = (np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int)
    cols = (np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int)
    return mask[np.ix_(rows, cols)]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of ``mask`` by a (2 * radius + 1)-square, within its bounds."""
    if radius < 1:
        return mask
    window = 2 * radius + 1
    padded = np.pad(mask, radius)
    rows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0).any(axis=-1)
    return np.asarray(
        np.lib.stride_tricks.sliding_window_view(rows, window, axis=1).any(axis=-1)
    )

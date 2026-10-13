from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from lymph_nodes.data.datasets.embedding import EmbeddingClassificationDataset
from lymph_nodes.data.datasets.meta_dataset import column_values


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ratiopath.openslide import OpenSlide

    from lymph_nodes.data.datasets.embedding import TileLabels
    from lymph_nodes.data.datasets.meta_dataset import FilterCondition
    from lymph_nodes.typedefs import SegmentationSample


class TileSegmentationDataset(EmbeddingClassificationDataset):
    """Tile images with per-patch targets, for SegmentationMetaArch.

    The tiles, labels, filters and per-slide structure of EmbeddingClassificationDataset
    — so StratifiedEpochSampler, the per-cohort metrics and PredictionMasks work alike —
    from a tiling run with coverage grids (configs/experiment/tiling/*_segmentation.yaml).
    A sample is instead:

    - the tile's RGB pixels, read from its slide: uint8 (H, W, 3),
    - its target grid: float32 (grid, grid), the coverage of each patch — per tile the
      maximum over the ``target_grids`` columns it has (OverlayGrid output; e.g.
      ``cytokeratin_grid`` for TMAs, ``annotation_grid`` for lymph nodes), 0 where it
      has none of them (e.g. a lymph node slide without an annotation mask, i.e. a
      negative one),
    - the usual meta, plus ``label``: the tile's TileLabels target, for tile metrics.

    ``grid`` is the side of the target grids (OverlayGrid's ``grid`` in the tiling),
    so that a batch none of whose tiles has a grid still gets targets of that shape.
    """

    def __init__(
        self,
        uris: Iterable[str],
        target_grids: Sequence[str],
        grid: int = 16,
        labels: TileLabels | None = None,
        tile_filters: Sequence[FilterCondition] = (),
        slide_filters: Sequence[FilterCondition] = (),
        exclude_unannotated_positive_slides: bool = False,
    ) -> None:
        self.target_grids = list(target_grids)
        self.grid = grid
        self._open_slides: dict[int, OpenSlide] = {}
        super().__init__(
            uris,
            labels=labels,
            tile_filters=tile_filters,
            slide_filters=slide_filters,
            exclude_unannotated_positive_slides=exclude_unannotated_positive_slides,
        )
        # Per position in .slides (= in .datasets): where and how to read its tiles.
        self._slide_paths = [str(path) for path in column_values(self.slides, "path")]
        self._slide_levels = column_values(self.slides, "level").astype(int)
        self._tile_extents = np.stack(
            [
                column_values(self.slides, "tile_extent_x"),
                column_values(self.slides, "tile_extent_y"),
            ],
            axis=1,
        ).astype(int)

    def __getstate__(self) -> dict[str, Any]:
        # Open slide handles stay in the process that opened them (DataLoader workers
        # open their own).
        return {**self.__dict__, "_open_slides": {}}

    def __getitems__(self, indices: Sequence[int]) -> list[SegmentationSample]:  # type: ignore[override]
        index = self._get_batch_index()
        batch = np.asarray(indices)
        grids = self._grids(index.rows[batch])
        samples: list[SegmentationSample] = []
        for i, position in enumerate(batch.tolist()):
            slide = int(index.slide_of[position])
            tile_x, tile_y = int(index.tile_x[position]), int(index.tile_y[position])
            meta = {
                "name": index.names[slide],
                "cohort": index.cohorts[slide],
                "slide": slide,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "ambiguous": bool(index.ambiguous[position]),
                "label": float(index.target[position]),
            }
            samples.append((self._read_tile(slide, tile_x, tile_y), grids[i], meta))
        return samples

    def _read_tile(self, slide: int, tile_x: int, tile_y: int) -> np.ndarray:
        from ratiopath.openslide import OpenSlide

        if slide not in self._open_slides:
            self._open_slides[slide] = OpenSlide(self._slide_paths[slide])
        extent_x, extent_y = self._tile_extents[slide]
        return self._open_slides[slide].read_tile(
            tile_x, tile_y, extent_x, extent_y, self._slide_levels[slide]
        )

    def _grids(self, rows: np.ndarray) -> np.ndarray:
        """Target grids of ``rows`` of the underlying table: float32 (len(rows), grid, grid)."""
        out: np.ndarray | None = None
        for name in self.target_grids:
            if name not in self.tiles.column_names:
                continue
            values, valid = self._take_grid(name, rows)
            grids = np.where(valid[:, np.newaxis], values, 0).astype(np.float32) / 255
            out = grids if out is None else np.maximum(out, grids)
        if out is None:
            raise ValueError(
                f"None of the target grids {self.target_grids} is a column of the tiles"
            )
        return out.reshape(len(rows), self.grid, self.grid)

    def _take_grid(self, name: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One grid column at ``rows`` (as in _embeddings, chunk by chunk).

        Returns the values (len(rows), cells) and which rows have any (not null).
        """
        column = self.tiles.data.column(name)
        offsets = np.cumsum([0, *(len(chunk) for chunk in column.chunks)])
        chunk_of = np.searchsorted(offsets, rows, side="right") - 1
        values = np.zeros((len(rows), self.grid * self.grid), dtype=np.uint8)
        valid = np.zeros(len(rows), dtype=bool)
        for chunk in np.unique(chunk_of):
            selected = np.flatnonzero(chunk_of == chunk)
            taken = column.chunk(int(chunk)).take(
                pa.array(rows[selected] - offsets[chunk])
            )
            values[selected], valid[selected] = _grid_rows(taken, self.grid * self.grid)
        return values, valid


def _grid_rows(array: pa.Array, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """The rows of a grid column: uint8 (len(array), cells) (0 where null), and validity.

    OverlayGrid writes a list, but Ray Data may store such a column as its tensor
    extension — a struct of the row's values (``data``) and shape — in the parquet
    (e.g. after a block with numpy batches); both are read here, as are fixed-size
    lists (earlier OverlayGrid output).
    """
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    valid = array.is_valid().to_numpy(zero_copy_only=False)
    if pa.types.is_struct(array.type):
        array = pc.struct_field(array, "data")
    # Flattening skips null rows: the values of the valid ones, in order.
    flat = pc.list_flatten(array).to_numpy(zero_copy_only=False)
    n_valid = int(valid.sum())
    if n_valid * cells != len(flat):
        raise ValueError(f"Grid rows with other than {cells} values")
    values = np.zeros((len(array), cells), dtype=np.uint8)
    values[valid] = flat.reshape(n_valid, cells)
    return values, valid

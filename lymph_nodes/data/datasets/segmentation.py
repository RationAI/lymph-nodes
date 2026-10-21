from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from lymph_nodes.data.datasets.embedding import (
    CANCER,
    HEALTHY,
    HEALTHY_BROWNISH,
    HEALTHY_EXTERNAL,
    EmbeddingClassificationDataset,
)
from lymph_nodes.data.datasets.meta_dataset import column_values
from preprocessing.white_balance import WHITE_POINT_COLUMNS, apply_white_point


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ratiopath.openslide import OpenSlide

    from lymph_nodes.data.augmentations import CellPasting, TileAugmentation
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
    - the usual meta, plus ``label``: the tile's TileLabels target, for tile metrics;
      with ``outline_grids`` (and ``outline_labels``, see EmbeddingClassificationDataset)
      also ``outline_grid``, a second target grid for a second set of patch metrics
      (SegmentationMetaArch), left out when the tiles have none of its columns.

    ``grid`` is the side of the target grids (OverlayGrid's ``grid`` in the tiling),
    so that a batch none of whose tiles has a grid still gets targets of that shape.

    Each DataLoader worker keeps at most ``max_open_slides`` slides open (least
    recently used closed first), all sharing one tile cache of ``cache_mib`` MiB:
    sampling across hundreds of lymph node slides, a handle per slide each with its own
    cache grew past the job's memory.

    For training (lymph_nodes.data.augmentations): ``cell_pasting`` pastes tumour cells
    cut from the cancer tiles of its ``donor_cohorts`` (TMAs) into the negative tiles of
    the other cohorts (lymph nodes; the healthy, healthy_brownish and healthy_external
    partitions), and the brown spots of other healthy_brownish tiles as negatives;
    ``augmentation`` then changes every tile's orientation, stain and focus, its target
    grid alike. Each worker draws from its own generator.
    """

    def __init__(
        self,
        uris: Iterable[str],
        target_grids: Sequence[str],
        grid: int = 16,
        max_open_slides: int = 64,
        cache_mib: int = 256,
        labels: TileLabels | None = None,
        tile_filters: Sequence[FilterCondition] = (),
        slide_filters: Sequence[FilterCondition] = (),
        exclude_unannotated_positive_slides: bool = False,
        ignore_unannotated_positive_slides: bool = False,
        outline_labels: TileLabels | None = None,
        outline_grids: Sequence[str] = (),
        augmentation: TileAugmentation | None = None,
        cell_pasting: CellPasting | None = None,
    ) -> None:
        self.augmentation = augmentation
        self.cell_pasting = cell_pasting
        self._generator: np.random.Generator | None = None
        self.target_grids = list(target_grids)
        self.outline_grids = list(outline_grids)
        self.grid = grid
        self.max_open_slides = max_open_slides
        self.cache_mib = cache_mib
        self._open_slides: OrderedDict[int, OpenSlide] = OrderedDict()
        self._slide_cache: Any = None
        super().__init__(
            uris,
            labels=labels,
            tile_filters=tile_filters,
            slide_filters=slide_filters,
            exclude_unannotated_positive_slides=exclude_unannotated_positive_slides,
            ignore_unannotated_positive_slides=ignore_unannotated_positive_slides,
            outline_labels=outline_labels,
        )
        if self.outline_grids and not any(
            g in self.tiles.column_names for g in self.outline_grids
        ):
            print(f"outline_grids left out: none of {self.outline_grids} in the tiles")
            self.outline_grids = []
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
        # The slides' white points, if the slides table has them: tiles of underexposed
        # scans are corrected as in the tiling (preprocessing.white_balance).
        self._white_points = (
            np.stack(
                [
                    column_values(self.slides, c).astype(float)
                    for c in WHITE_POINT_COLUMNS
                ],
                axis=1,
            )
            if all(c in self.slides.column_names for c in WHITE_POINT_COLUMNS)
            else None
        )
        self._mpp = (
            column_values(self.slides, "mpp_x").astype(float)
            if "mpp_x" in self.slides.column_names
            else np.full(len(self._slide_paths), 0.5)
        )
        if cell_pasting is not None:
            self._pasting_pools(cell_pasting)

    def _pasting_pools(self, cell_pasting: CellPasting) -> None:
        """Positions (global indices) of the donors, recipients and brown donors."""
        index = self._get_batch_index()
        partitions = np.concatenate(
            [dataset.partition_labels() for dataset in self.datasets]  # type: ignore[attr-defined]
        )
        donor_cohort = np.isin(np.asarray(index.cohorts), cell_pasting.donor_cohorts)[
            index.slide_of
        ]
        negative = np.isin(partitions, [HEALTHY, HEALTHY_BROWNISH, HEALTHY_EXTERNAL])
        self._donors = np.flatnonzero(donor_cohort & (partitions == CANCER))
        self._recipients = ~donor_cohort & negative
        self._brown_donors = np.flatnonzero(
            ~donor_cohort & (partitions == HEALTHY_BROWNISH)
        )
        print(
            f"cell pasting: {len(self._donors)} donor tiles, "
            f"{int(self._recipients.sum())} recipients, "
            f"{len(self._brown_donors)} brown donors"
        )
        if len(self._donors) == 0:
            print("cell pasting left out: no donor tiles")
            self.cell_pasting = None

    def __getstate__(self) -> dict[str, Any]:
        # Open slide handles stay in the process that opened them (DataLoader workers
        # open their own).
        return {
            **self.__dict__,
            "_open_slides": OrderedDict(),
            "_slide_cache": None,
            "_generator": None,
        }

    def _rng(self) -> np.random.Generator:
        """The worker's generator, seeded by its torch seed (distinct per worker)."""
        if self._generator is None:
            import torch

            info = torch.utils.data.get_worker_info()
            seed = info.seed if info is not None else torch.initial_seed()
            self._generator = np.random.default_rng(seed % 2**32)
        return self._generator

    def __getitems__(self, indices: Sequence[int]) -> list[SegmentationSample]:  # type: ignore[override]
        index = self._get_batch_index()
        batch = np.asarray(indices)
        grids = self._grids(index.rows[batch])
        outline = (
            self._grids(index.rows[batch], self.outline_grids)
            if self.outline_grids
            else None
        )
        # The tiling's tissue (per patch), where the augmentations need it.
        tissue = (
            self._take_grid("tissue_grid", index.rows[batch])
            if (self.augmentation or self.cell_pasting)
            and "tissue_grid" in self.tiles.column_names
            else None
        )
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
                "ignore": index.ignores[slide],
                "label": float(index.target[position]),
            }
            if outline is not None:
                meta["outline_grid"] = outline[i]
            if index.outline is not None:
                meta["outline_label"] = float(index.outline[position])
            tile, grid = self._read_tile(slide, tile_x, tile_y), grids[i]
            pixels = None
            if tissue is not None and tissue[1][i]:
                cells = tissue[0][i].reshape(self.grid, self.grid) >= 128
                scale = tile.shape[0] // self.grid
                pixels = np.repeat(np.repeat(cells, scale, axis=0), scale, axis=1)
            if self.cell_pasting is not None and self._recipients[position]:
                tile, grid = self._paste(tile, grid, slide, pixels)
                if grid.max() > 0:  # cells pasted
                    meta["label"] = 1.0
            if self.augmentation is not None:
                tile, grid = self.augmentation(tile, grid, self._rng(), pixels)
            samples.append((tile, grid, meta))
        return samples

    def _paste(
        self,
        tile: np.ndarray,
        grid: np.ndarray,
        slide: int,
        tissue: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``cell_pasting`` on a recipient tile, reading the donor tiles it needs."""
        assert self.cell_pasting is not None
        rng = self._rng()
        donor = brown = None
        if rng.random() < self.cell_pasting.p:
            donor = self._read_position(int(rng.choice(self._donors)))
        if len(self._brown_donors) and rng.random() < self.cell_pasting.negative_p:
            brown = self._read_position(int(rng.choice(self._brown_donors)))
        if donor is None and brown is None:
            return tile, grid
        return self.cell_pasting(
            tile,
            grid,
            float(self._mpp[slide]),
            rng,
            donor=donor,
            brown=brown,
            tissue=tissue,
        )

    def _read_position(self, position: int) -> np.ndarray:
        """The tile at a global index."""
        index = self._get_batch_index()
        return self._read_tile(
            int(index.slide_of[position]),
            int(index.tile_x[position]),
            int(index.tile_y[position]),
        )

    def _read_tile(self, slide: int, tile_x: int, tile_y: int) -> np.ndarray:
        from ratiopath.openslide import OpenSlide

        if slide in self._open_slides:
            self._open_slides.move_to_end(slide)
        else:
            if len(self._open_slides) >= self.max_open_slides:
                self._open_slides.popitem(last=False)[1].close()
            handle = OpenSlide(self._slide_paths[slide])
            self._share_cache(handle)
            self._open_slides[slide] = handle
        extent_x, extent_y = self._tile_extents[slide]
        tile = self._open_slides[slide].read_tile(
            tile_x, tile_y, extent_x, extent_y, self._slide_levels[slide]
        )
        white = None if self._white_points is None else self._white_points[slide]
        return apply_white_point(
            tile, white if white is not None and np.isfinite(white).all() else None
        )

    def _share_cache(self, handle: OpenSlide) -> None:
        """Points the handle at the worker's shared tile cache (OpenSlide >= 4.0 only)."""
        from openslide import OpenSlideCache, OpenSlideVersionError

        try:
            if self._slide_cache is None:
                self._slide_cache = OpenSlideCache(self.cache_mib * 2**20)
            handle.set_cache(self._slide_cache)
        except OpenSlideVersionError:
            pass  # an older library: only max_open_slides bounds the memory

    def _grids(
        self, rows: np.ndarray, names: Sequence[str] | None = None
    ) -> np.ndarray:
        """Target grids of ``rows`` of the underlying table: float32 (len(rows), grid, grid).

        The per-cell maximum over the ``names`` columns (``target_grids`` by default).
        """
        out: np.ndarray | None = None
        names = self.target_grids if names is None else names
        for name in names:
            if name not in self.tiles.column_names:
                continue
            values, valid = self._take_grid(name, rows)
            grids = np.where(valid[:, np.newaxis], values, 0).astype(np.float32) / 255
            out = grids if out is None else np.maximum(out, grids)
        if out is None:
            raise ValueError(
                f"None of the target grids {list(names)} is a column of the tiles"
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

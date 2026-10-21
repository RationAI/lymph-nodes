from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import pyarrow as pa
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.meta_dataset import MetaDataset, column_values
from lymph_nodes.typedefs import EmbeddingSample


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from datasets import Dataset as HFDataset

    from lymph_nodes.data.datasets.meta_dataset import FilterCondition


# Partition ids used both for the per-tile target (0/1/2 collapsed to healthy/cancer in
# TileLabels.targets) and for lymph_nodes.data.samplers.stratified_sampler's rebalancing.
# Keep in sync with that module's PARTITION_NAMES.
HEALTHY, HEALTHY_BROWNISH, CANCER, HEALTHY_TMA, HEALTHY_EXTERNAL = 0, 1, 2, 3, 4


def _coverage(tiles: HFDataset, column: str) -> np.ndarray | None:
    """A coverage column as floats (missing mask -> NaN), or None if no run has it."""
    if column not in tiles.column_names:
        return None
    return column_values(tiles, column).astype(np.float64)


class TileLabels:
    """How a tile's label is derived from the coverage columns of the tiling pipeline.

    There is no stored "label" column. A tile is cancer if any column in ``positive``
    exceeds its threshold — e.g. the pathologist's annotation (``annotation_coverage``)
    or the cytokeratin epithelium overlay (``cytokeratin_coverage``). Otherwise it is
    "healthy brownish" if ``brownish_column`` exceeds ``brownish_threshold``: tissue that
    resembles metastasis on staining alone, split out so the StratifiedEpochSampler
    can oversample it as a static stand-in for hard-negative mining. Otherwise it is
    "healthy TMA" if ``tma_column`` — a column only TMA tiles have, e.g.
    ``cytokeratin_coverage`` — is defined for it: negative TMA tissue, split out so the
    sampler can draw it in its own ratio next to the lymph node negatives. Everything
    else is healthy. Non-cancer tiles of the slides of ``external_cohort`` (their
    ``cohort_id``) are "healthy external" instead, whatever their staining: tissue from
    another site, split out so the sampler can draw it in its own ratio rather than let
    it flood the partitions of the main cohort — e.g. FN Brno, where most of the
    negative tissue is brownish (~70% of its tiles, against ~2% at MMCI).

    A column absent from the tiles (e.g. no cytokeratin masks for lymph nodes) or
    undefined for a slide (NaN) never makes a tile positive or brownish.

    The training target is the 0/1 cancer label, unless ``soft_target`` names a
    coverage column: then a tile's target is that coverage (e.g. the fraction of its
    pixels that are cytokeratin positive), so the predicted probability estimates it.
    Ambiguous tiles get an in-between target instead of being forced to 0 or 1, and no
    target pushes the logits to infinity. Tiles where the column is NaN keep their 0/1
    label. Metrics still count a tile as cancer when its target is above 0.5 (MetaArch).

    ``ambiguous`` maps a coverage column to ``[low, high]``: tiles with ``low < value <=
    high`` are borderline, e.g. lymph node tiles only touching an annotation
    (``annotation_coverage: [0.0, 0.25]``). They keep their label and are evaluated as
    usual, but MetaArch also reports every metric without them (``<cohort>/unambiguous/``).
    With ``ambiguous_on_positive_slides``, the negative tiles of positive slides (the
    slides' ``tumor``) are ambiguous too: the pathologist may have left an ITC there
    unannotated, so a model flagging it is not necessarily wrong — the unambiguous
    metrics then take their negatives from negative slides only.

    Deliberately a plain class rather than a dataclass, like FilterCondition.
    """

    def __init__(
        self,
        positive: Mapping[str, float],
        brownish_column: str | None = None,
        brownish_threshold: float = 0.0,
        soft_target: str | None = None,
        tma_column: str | None = None,
        ambiguous: Mapping[str, Sequence[float]] | None = None,
        external_cohort: str | None = None,
        ambiguous_on_positive_slides: bool = False,
    ) -> None:
        self.positive = dict(positive)
        self.ambiguous_on_positive_slides = ambiguous_on_positive_slides
        self.external_cohort = external_cohort
        self.brownish_column = brownish_column
        self.brownish_threshold = brownish_threshold
        self.soft_target = soft_target
        self.tma_column = tma_column
        self.ambiguous = {column: (float(low), float(high)) for column, (low, high) in (ambiguous or {}).items()}

    def partition(self, tiles: HFDataset, slides: HFDataset | None = None) -> np.ndarray:
        """Per-tile partition id: HEALTHY, HEALTHY_BROWNISH, HEALTHY_TMA, HEALTHY_EXTERNAL or CANCER.

        ``slides`` (with ``id`` and ``cohort_id``) is needed for ``external_cohort``.
        """
        cancer = self._cancer(tiles)

        brownish = np.zeros(len(tiles), dtype=bool)
        if self.brownish_column is not None:
            values = _coverage(tiles, self.brownish_column)
            if values is not None:
                brownish = values > self.brownish_threshold

        tma = np.zeros(len(tiles), dtype=bool)
        if self.tma_column is not None:
            values = _coverage(tiles, self.tma_column)
            if values is not None:
                tma = ~np.isnan(values)

        external = np.zeros(len(tiles), dtype=bool)
        if self.external_cohort is not None:
            if slides is None or "cohort_id" not in slides.column_names:
                raise ValueError("external_cohort needs slides with a cohort_id column")
            ids = column_values(slides, "id")[column_values(slides, "cohort_id") == self.external_cohort]
            external = np.isin(column_values(tiles, "slide_id"), ids)

        labels = np.full(len(tiles), HEALTHY, dtype=np.int64)
        labels[brownish & ~cancer] = HEALTHY_BROWNISH
        labels[tma & ~cancer] = HEALTHY_TMA
        labels[external & ~cancer] = HEALTHY_EXTERNAL
        labels[cancer] = CANCER
        return labels

    def _cancer(self, tiles: HFDataset) -> np.ndarray:
        """Per-tile flag: positive by any column of ``positive``."""
        cancer = np.zeros(len(tiles), dtype=bool)
        for column, threshold in self.positive.items():
            values = _coverage(tiles, column)
            if values is not None:
                cancer |= values > threshold
        return cancer

    def ambiguous_mask(self, tiles: HFDataset, slides: HFDataset | None = None) -> np.ndarray:
        """Per-tile flag: borderline by any ``ambiguous`` range (NaN never is).

        With ``ambiguous_on_positive_slides``, also every negative tile of a positive
        slide; ``slides`` (with ``id`` and ``tumor``) is needed for that.
        """
        ambiguous = np.zeros(len(tiles), dtype=bool)
        for column, (low, high) in self.ambiguous.items():
            values = _coverage(tiles, column)
            if values is not None:
                ambiguous |= (values > low) & (values <= high)
        if self.ambiguous_on_positive_slides:
            if slides is None or "tumor" not in slides.column_names:
                raise ValueError("ambiguous_on_positive_slides needs slides with a tumor column")
            # TMA slides have no status (None / NaN): never positive.
            tumor = column_values(slides, "tumor")
            positive = np.array([bool(t) and t == t for t in tumor], dtype=bool)
            ids = column_values(slides, "id")[positive]
            on_positive = np.isin(column_values(tiles, "slide_id"), ids)
            ambiguous |= on_positive & ~self._cancer(tiles)
        return ambiguous

    def targets(self, tiles: HFDataset, partitions: np.ndarray) -> np.ndarray:
        """Per-tile training target in [0, 1], given the tiles' ``partition``."""
        targets = (partitions == CANCER).astype(np.float32)
        if self.soft_target is not None:
            values = _coverage(tiles, self.soft_target)
            if values is not None:
                defined = ~np.isnan(values)
                targets[defined] = np.clip(values[defined], 0.0, 1.0)
        return targets


class TileEmbeddingClassificationDataset(Dataset[EmbeddingSample]):
    """One slide's tiles, with a cancer target derived from overlay coverage.

    ``partitions`` holds each tile's TileLabels partition id, ``targets`` its target
    (TileLabels.targets), ``ambiguous`` its TileLabels.ambiguous_mask flag and ``rows``
    its row in the parent EmbeddingClassificationDataset's table, all in ``tiles``
    order. Each sample carries, besides the slide's ``name``, its ``cohort`` (the
    slide's cohort_id) and ``ambiguous`` flag, for the per-cohort metrics, and where the
    tile lies — ``slide``, the slide's position in the parent's ``slides``, and
    ``tile_x``/``tile_y`` — for the PredictionMasks callback.
    """

    def __init__(
        self,
        name: str,
        cohort: str,
        slide: int,
        tiles: HFDataset,
        partitions: np.ndarray,
        targets: np.ndarray,
        ambiguous: np.ndarray,
        rows: np.ndarray,
        input_column: str = "embedding",
        ignore: bool = False,
        outline_targets: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.input_column = input_column
        self.ignore = ignore
        self.outline_targets = outline_targets
        self.name = name
        self.cohort = cohort
        self.slide = slide
        self.tiles = tiles
        self.targets = targets
        self.ambiguous = ambiguous
        self.rows = rows
        self._partitions = partitions

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> EmbeddingSample:
        tile = self.tiles[idx]
        meta = {
            "name": self.name,
            "cohort": self.cohort,
            "slide": self.slide,
            "tile_x": int(tile["tile_x"]),
            "tile_y": int(tile["tile_y"]),
            "ambiguous": bool(self.ambiguous[idx]),
            "ignore": self.ignore,
        }
        if self.outline_targets is not None:
            meta["outline_label"] = float(self.outline_targets[idx])
        value = np.atleast_1d(np.asarray(tile[self.input_column], dtype=np.float32))
        return np.nan_to_num(value), float(self.targets[idx]), meta

    def partition_labels(self) -> np.ndarray:
        """Per-tile group id (HEALTHY, HEALTHY_BROWNISH, HEALTHY_TMA, HEALTHY_EXTERNAL or CANCER), for the sampler."""
        return self._partitions


class _BatchIndex(NamedTuple):
    """Per global index of an EmbeddingClassificationDataset, plus per-slide values."""

    rows: np.ndarray  # row in the flat tile table
    target: np.ndarray
    ambiguous: np.ndarray
    slide_of: np.ndarray  # position in .datasets (= in .slides)
    tile_x: np.ndarray
    tile_y: np.ndarray
    names: list[str]
    cohorts: list[str]
    ignores: list[bool]  # per slide: tile metrics skip it (ignore_unannotated_positive_slides)
    outline: np.ndarray | None  # outline_labels targets, if any


class EmbeddingClassificationDataset(MetaDataset[EmbeddingSample]):
    """Tile embeddings of every slide in ``uris`` with labels from ``labels``.

    ``exclude_unannotated_positive_slides`` drops slides labelled positive (``tumor``)
    that have no annotation mask at all: their metastases aren't outlined, so every
    tile would be labelled healthy. Needed wherever lymph node slides are evaluated
    by ``annotation_coverage``, since most positive slides have no annotation.
    ``outline_labels``, a second TileLabels (the pathologists' outlines, when ``labels``
    are the refined ones of preprocessing/refined_annotation_masks.py), puts a second
    target in the samples' meta (``outline_label``) for a second set of metrics
    (MetaArch); left out when a column it names is missing from the tiles.

    ``ignore_unannotated_positive_slides`` keeps those slides instead but flags their
    samples (meta ``ignore``): tile-level metrics skip them, while slide-level ones
    (lymph_nodes.callbacks.SlideMetrics) need every slide with its status. Excluding
    wins if both are set.

    ``input_column`` is what a sample's input is read from: the embedding by default,
    or a scalar column as a one-element input (missing values as 0) — e.g.
    ``brownish_coverage`` for the brown-staining baseline (configs/experiment/
    training/brown_baseline.yaml).
    """

    def __init__(
        self,
        uris: Iterable[str],
        labels: TileLabels | None = None,
        tile_filters: Sequence[FilterCondition] = (),
        slide_filters: Sequence[FilterCondition] = (),
        exclude_unannotated_positive_slides: bool = False,
        input_column: str = "embedding",
        ignore_unannotated_positive_slides: bool = False,
        outline_labels: TileLabels | None = None,
    ) -> None:
        self.input_column = input_column
        self.outline_labels = outline_labels
        self.ignore_unannotated_positive_slides = ignore_unannotated_positive_slides
        self._unannotated_positive: set[str] = set()
        self._batch_index: _BatchIndex | None = None
        self._chunk_offsets: np.ndarray | None = None
        # Set before super().__init__, which builds the per-slide datasets.
        self.labels = labels or TileLabels(
            positive={"annotation_coverage": 0.0, "cytokeratin_coverage": 0.0},
            brownish_column="brownish_coverage",
            brownish_threshold=0.2,
        )
        self.exclude_unannotated_positive_slides = exclude_unannotated_positive_slides
        super().__init__(uris, tile_filters=tile_filters, slide_filters=slide_filters)

    def _select_slides(self, slides: HFDataset, tiles: HFDataset) -> HFDataset:
        if not (self.exclude_unannotated_positive_slides or self.ignore_unannotated_positive_slides):
            return slides
        annotation = _coverage(tiles, "annotation_coverage")
        annotated = (
            np.unique(column_values(tiles, "slide_id")[~np.isnan(annotation)])
            if annotation is not None
            else np.array([])
        )
        ids = column_values(slides, "id")
        keep = ~column_values(slides, "tumor").astype(bool) | np.isin(ids, annotated)
        if self.exclude_unannotated_positive_slides:
            return slides.select(np.flatnonzero(keep)).flatten_indices()
        self._unannotated_positive = {str(i) for i in ids[~keep]}
        return slides

    def generate_datasets(self) -> Iterable[TileEmbeddingClassificationDataset]:
        # Labels for the whole table at once: a column of the flat table reads just that
        # column, whereas reading one through a per-slide view (filter_tiles_by_slide)
        # gathers the slide's full rows, embeddings included.
        partitions = self.labels.partition(self.tiles, self.slides)
        targets = self.labels.targets(self.tiles, partitions)
        ambiguous = self.labels.ambiguous_mask(self.tiles, self.slides)
        outline = self._outline_targets()
        has_cohort = "cohort_id" in self.slides.column_names
        no_rows = np.array([], dtype=np.int64)
        for position, slide in enumerate(self.slides):
            index = self._slide_id_to_indices.get(slide["id"])
            rows = index.values.to_numpy() if index is not None else no_rows
            yield TileEmbeddingClassificationDataset(
                name=slide["slide_name"],
                cohort=str(slide["cohort_id"]) if has_cohort else "unknown",
                slide=position,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                partitions=partitions[rows],
                targets=targets[rows],
                ambiguous=ambiguous[rows],
                rows=rows,
                input_column=self.input_column,
                ignore=str(slide["id"]) in self._unannotated_positive,
                outline_targets=outline[rows] if outline is not None else None,
            )

    def _outline_targets(self) -> np.ndarray | None:
        """Targets by ``outline_labels``; None without them, or if the tiles lack all their columns.

        A column missing from these tiles but not all of them is fine, as for ``labels``
        (e.g. no TMAs, so no cytokeratin coverage): it never makes a tile positive.
        """
        if self.outline_labels is None:
            return None
        if not any(c in self.tiles.column_names for c in self.outline_labels.positive):
            print(f"outline_labels left out: none of {list(self.outline_labels.positive)} in the tiles")
            return None
        partitions = self.outline_labels.partition(self.tiles, self.slides)
        return self.outline_labels.targets(self.tiles, partitions)

    def __getitems__(self, indices: Sequence[int]) -> list[EmbeddingSample]:
        """A whole batch with one read of the embedding column.

        The DataLoader calls this instead of ``__getitem__`` per sample when it exists.
        Fetching rows one at a time through the per-slide views costs a lookup and a
        conversion per tile, which leaves the loader workers, not the GPU, as the
        bottleneck.
        """
        index = self._get_batch_index()
        batch = np.asarray(indices)
        values = self._embeddings(index.rows[batch])
        slides = index.slide_of[batch].tolist()
        outline = index.outline[batch].tolist() if index.outline is not None else None
        return [
            (
                values[i],
                target,
                {
                    "name": index.names[slide],
                    "cohort": index.cohorts[slide],
                    "slide": slide,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "ambiguous": ambiguous,
                    "ignore": index.ignores[slide],
                    **({"outline_label": outline[i]} if outline is not None else {}),
                },
            )
            for i, (target, slide, tile_x, tile_y, ambiguous) in enumerate(
                zip(
                    index.target[batch].tolist(),
                    slides,
                    index.tile_x[batch].tolist(),
                    index.tile_y[batch].tolist(),
                    index.ambiguous[batch].tolist(),
                    strict=True,
                )
            )
        ]

    def _embeddings(self, rows: np.ndarray) -> np.ndarray:
        """Inputs (``input_column``) of ``rows`` of the underlying table, as one (len(rows), dim) array.

        Taken chunk by chunk: ``ChunkedArray.take`` on the (large_list) embedding column
        concatenates every chunk first — the whole column, materialized in each loader
        worker. ``rows`` index self.tiles.data, like _slide_id_to_indices.
        """
        column = self.tiles.data.column(self.input_column)
        if self._chunk_offsets is None:
            self._chunk_offsets = np.cumsum([0, *(len(chunk) for chunk in column.chunks)])
        chunk_of = np.searchsorted(self._chunk_offsets, rows, side="right") - 1

        out: np.ndarray | None = None
        for chunk in np.unique(chunk_of):
            selected = np.flatnonzero(chunk_of == chunk)
            local = pa.array(rows[selected] - self._chunk_offsets[chunk])
            taken = column.chunk(int(chunk)).take(local)
            if pa.types.is_floating(taken.type) or pa.types.is_integer(taken.type):
                values = np.nan_to_num(taken.to_numpy(zero_copy_only=False).astype(np.float32))
            else:
                values = taken.flatten().to_numpy(zero_copy_only=False)
            values = values.reshape(len(selected), -1)
            if out is None:
                out = np.empty((len(rows), values.shape[1]), dtype=values.dtype)
            out[selected] = values
        return out if out is not None else np.empty((0, 0), dtype=np.float32)

    def _get_batch_index(self) -> _BatchIndex:
        if self._batch_index is None:
            slides = cast("list[TileEmbeddingClassificationDataset]", self.datasets)
            rows = np.concatenate([slide.rows for slide in slides])
            self._batch_index = _BatchIndex(
                rows=rows,
                target=np.concatenate([slide.targets for slide in slides]),
                ambiguous=np.concatenate([slide.ambiguous for slide in slides]),
                slide_of=np.repeat(np.arange(len(slides)), [len(slide) for slide in slides]),
                # rows index self.tiles.data, as in _embeddings
                tile_x=self.tiles.data.column("tile_x").to_numpy()[rows],
                tile_y=self.tiles.data.column("tile_y").to_numpy()[rows],
                names=[slide.name for slide in slides],
                cohorts=[slide.cohort for slide in slides],
                ignores=[slide.ignore for slide in slides],
                outline=(
                    np.concatenate([slide.outline_targets for slide in slides])
                    if slides and slides[0].outline_targets is not None
                    else None
                ),
            )
        return self._batch_index

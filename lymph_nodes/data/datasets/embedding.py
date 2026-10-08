from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.meta_dataset import MetaDataset, column_values
from lymph_nodes.typedefs import EmbeddingSample


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from datasets import Dataset as HFDataset

    from lymph_nodes.data.datasets.meta_dataset import FilterCondition


# Partition ids used both for the per-tile label (0/1/2 collapsed to healthy/cancer in
# __getitem__) and for lymph_nodes.data.samplers.stratified_sampler's rebalancing.
# Keep in sync with that module's PARTITION_NAMES.
HEALTHY, HEALTHY_BROWNISH, CANCER = 0, 1, 2


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
    can oversample it as a static stand-in for hard-negative mining. Everything else
    is healthy.

    A column absent from the tiles (e.g. no cytokeratin masks for lymph nodes) or
    undefined for a slide (NaN) never makes a tile positive or brownish.

    Deliberately a plain class rather than a dataclass, like FilterCondition.
    """

    def __init__(
        self,
        positive: Mapping[str, float],
        brownish_column: str | None = None,
        brownish_threshold: float = 0.0,
    ) -> None:
        self.positive = dict(positive)
        self.brownish_column = brownish_column
        self.brownish_threshold = brownish_threshold

    def partition(self, tiles: HFDataset) -> np.ndarray:
        """Per-tile partition id: HEALTHY, HEALTHY_BROWNISH, or CANCER."""
        cancer = np.zeros(len(tiles), dtype=bool)
        for column, threshold in self.positive.items():
            values = _coverage(tiles, column)
            if values is not None:
                cancer |= values > threshold

        brownish = np.zeros(len(tiles), dtype=bool)
        if self.brownish_column is not None:
            values = _coverage(tiles, self.brownish_column)
            if values is not None:
                brownish = values > self.brownish_threshold

        labels = np.full(len(tiles), HEALTHY, dtype=np.int64)
        labels[brownish & ~cancer] = HEALTHY_BROWNISH
        labels[cancer] = CANCER
        return labels


class TileEmbeddingClassificationDataset(Dataset[EmbeddingSample]):
    """One slide's tiles, with a binary cancer label derived from overlay coverage.

    ``partitions`` holds each tile's TileLabels partition id and ``rows`` its row in the
    parent EmbeddingClassificationDataset's table, both in ``tiles`` order.
    """

    def __init__(self, name: str, tiles: HFDataset, partitions: np.ndarray, rows: np.ndarray) -> None:
        super().__init__()
        self.name = name
        self.tiles = tiles
        self.rows = rows
        self._partitions = partitions

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> EmbeddingSample:
        has_cancer = bool(self._partitions[idx] == CANCER)
        return self.tiles[idx]["embedding"], has_cancer, {"name": self.name}

    def partition_labels(self) -> np.ndarray:
        """Per-tile group id (HEALTHY, HEALTHY_BROWNISH, or CANCER), for the sampler."""
        return self._partitions


class EmbeddingClassificationDataset(MetaDataset[EmbeddingSample]):
    """Tile embeddings of every slide in ``uris`` with labels from ``labels``.

    ``exclude_unannotated_positive_slides`` drops slides labelled positive (``tumor``)
    that have no annotation mask at all: their metastases aren't outlined, so every
    tile would be labelled healthy. Needed wherever lymph node slides are evaluated
    by ``annotation_coverage``, since most positive slides have no annotation.
    """

    def __init__(
        self,
        uris: Iterable[str],
        labels: TileLabels | None = None,
        tile_filters: Sequence[FilterCondition] = (),
        slide_filters: Sequence[FilterCondition] = (),
        exclude_unannotated_positive_slides: bool = False,
    ) -> None:
        self._batch_index: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None = None
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
        if not self.exclude_unannotated_positive_slides:
            return slides
        annotation = _coverage(tiles, "annotation_coverage")
        annotated = (
            np.unique(column_values(tiles, "slide_id")[~np.isnan(annotation)])
            if annotation is not None
            else np.array([])
        )
        keep = ~column_values(slides, "tumor").astype(bool) | np.isin(column_values(slides, "id"), annotated)
        return slides.select(np.flatnonzero(keep)).flatten_indices()

    def generate_datasets(self) -> Iterable[TileEmbeddingClassificationDataset]:
        # Labels for the whole table at once: a column of the flat table reads just that
        # column, whereas reading one through a per-slide view (filter_tiles_by_slide)
        # gathers the slide's full rows, embeddings included.
        partitions = self.labels.partition(self.tiles)
        no_rows = np.array([], dtype=np.int64)
        for slide in self.slides:
            index = self._slide_id_to_indices.get(slide["id"])
            rows = index.values.to_numpy() if index is not None else no_rows
            yield TileEmbeddingClassificationDataset(
                name=slide["slide_name"],
                tiles=self.filter_tiles_by_slide(slide["id"]),
                partitions=partitions[rows],
                rows=rows,
            )

    def __getitems__(self, indices: Sequence[int]) -> list[EmbeddingSample]:
        """A whole batch with one read of the embedding column.

        The DataLoader calls this instead of ``__getitem__`` per sample when it exists.
        Fetching rows one at a time through the per-slide views costs a lookup and a
        conversion per tile, which leaves the loader workers, not the GPU, as the
        bottleneck.
        """
        rows, cancer, slide_of, names = self._get_batch_index()
        batch = np.asarray(indices)
        values = self._embeddings(rows[batch])
        return [
            (values[i], bool(cancer[index]), {"name": names[slide_of[index]]})
            for i, index in enumerate(batch.tolist())
        ]

    def _embeddings(self, rows: np.ndarray) -> np.ndarray:
        """Embeddings of ``rows`` of the underlying table, as one (len(rows), dim) array.

        Taken chunk by chunk: ``ChunkedArray.take`` on the (large_list) embedding column
        concatenates every chunk first — the whole column, materialized in each loader
        worker. ``rows`` index self.tiles.data, like _slide_id_to_indices.
        """
        column = self.tiles.data.column("embedding")
        if self._chunk_offsets is None:
            self._chunk_offsets = np.cumsum([0, *(len(chunk) for chunk in column.chunks)])
        chunk_of = np.searchsorted(self._chunk_offsets, rows, side="right") - 1

        out: np.ndarray | None = None
        for chunk in np.unique(chunk_of):
            selected = np.flatnonzero(chunk_of == chunk)
            local = pa.array(rows[selected] - self._chunk_offsets[chunk])
            values = column.chunk(int(chunk)).take(local).flatten().to_numpy(zero_copy_only=False)
            values = values.reshape(len(selected), -1)
            if out is None:
                out = np.empty((len(rows), values.shape[1]), dtype=values.dtype)
            out[selected] = values
        return out if out is not None else np.empty((0, 0), dtype=np.float32)

    def _get_batch_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Per global index: table row, cancer flag, slide position; plus slide names."""
        if self._batch_index is None:
            slides = cast("list[TileEmbeddingClassificationDataset]", self.datasets)
            self._batch_index = (
                np.concatenate([slide.rows for slide in slides]),
                np.concatenate([slide.partition_labels() == CANCER for slide in slides]),
                np.repeat(np.arange(len(slides)), [len(slide) for slide in slides]),
                [slide.name for slide in slides],
            )
        return self._batch_index

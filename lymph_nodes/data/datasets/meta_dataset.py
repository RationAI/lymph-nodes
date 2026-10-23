from __future__ import annotations

import contextlib
import operator as _op
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset as HFDataset
from datasets import Value, concatenate_datasets
from datasets.table import ConcatenationTable
from rationai.mlkit.data.datasets import MetaTiledSlides, load_dataset


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


def column_values(dataset: HFDataset, name: str) -> np.ndarray:
    """A whole column as one numpy array.

    In datasets 5, ``dataset[name]`` is a lazy Column: ``np.asarray`` on it converts
    row by row, reading every row's full record (embeddings included), which on
    millions of tiles thrashes through the memory-mapped tables. Slicing it reads the
    column in one go.
    """
    return np.asarray(dataset[name][:])


_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    ">": _op.gt, "gt": _op.gt,
    ">=": _op.ge, "ge": _op.ge,
    "<": _op.lt, "lt": _op.lt,
    "<=": _op.le, "le": _op.le,
    "==": _op.eq, "eq": _op.eq,
    "!=": _op.ne, "ne": _op.ne,
}


class FilterCondition:
    """A single column threshold to keep/drop rows by.

    ``op`` is either a comparison (">", ">=", "<", "<=", "==", "!=") applied against
    a scalar ``value``, "in" / "not in" applied against an iterable ``value`` (set
    membership), or "between" with ``value = [low, high]`` (``low < x <= high``).

    ``cohort`` restricts a tile filter to the tiles of slides whose ``cohort_id`` is
    that cohort; tiles of other cohorts are kept regardless. E.g.
    ``FilterCondition("cytokeratin_coverage", "<=", 0.5, negate=True, cohort="mmci_tmas")``
    keeps only cytokeratin-positive TMA tiles and leaves lymph node tiles untouched.

    Example: ``FilterCondition("tissue_coverage", ">", 0.2)``.

    ``negate=True`` keeps rows that *fail* the comparison instead of ones that pass —
    useful for optional (non-``mandatory``) coverage columns, which are ``NaN`` rather
    than 0 for slides with no mask at all. ``NaN > 0`` is already ``False`` in numpy,
    so "defined and > threshold" (the condition you want to *drop* on) is just
    ``column > value`` directly; ``negate=True`` keeps everything that condition
    doesn't match — ``NaN`` (undefined → keep) and ``<= value`` (defined, below
    threshold → keep) alike — without special-casing ``NaN`` at all. E.g.
    ``FilterCondition("ignore_coverage", ">", 0, negate=True)`` drops a tile only if
    ``ignore_coverage`` is defined *and* positive.

    Deliberately a plain class, not a ``@dataclass``: Hydra's recursive
    ``instantiate()`` does not construct a dataclass-typed ``_target_`` when it is
    nested inside a list field (e.g. a YAML ``tile_filters:`` list) — it treats it as
    a structured-config schema instead and leaves it as a ``DictConfig``, silently
    skipping instantiation. A plain class with the same shape does not hit that path.
    """

    def __init__(
        self, column: str, op: str, value: Any, negate: bool = False, cohort: str | None = None
    ) -> None:
        self.column = column
        self.op = op
        self.value = value
        self.negate = negate
        self.cohort = cohort

    def mask(self, dataset: HFDataset) -> np.ndarray:
        column = column_values(dataset, self.column)
        if column.dtype == object and self.op not in ("in", "not in"):
            # A column that is null throughout a run comes back as Python None values
            # rather than NaN; as floats they compare like any other missing coverage.
            with contextlib.suppress(TypeError, ValueError):
                column = column.astype(np.float64)
        if self.op in ("in", "not in"):
            is_in = np.isin(column, list(self.value))
            result = is_in if self.op == "in" else ~is_in
        elif self.op == "between":
            low, high = self.value
            result = (column > low) & (column <= high)
        else:
            result = _COMPARISONS[self.op](column, self.value)
        return ~result if self.negate else result


class RandomSubset:
    """Keeps a fixed random ``fraction`` of the rows, usable among tile filters.

    E.g. to validate on a tenth of a large cohort's tiles during ``fit``:
    ``RandomSubset(0.1, cohort="mmci_ihc_2023")`` (``cohort`` scopes it like
    FilterCondition's: tiles of other cohorts are kept). The subset is drawn with
    ``seed`` over the rows in table order, so the same data gives the same subset.
    """

    def __init__(self, fraction: float, cohort: str | None = None, seed: int = 0) -> None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction
        self.cohort = cohort
        self.seed = seed
        self.column = f"<random {fraction:.0%}>"  # for _select's messages

    def mask(self, dataset: HFDataset) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(dataset)) < self.fraction


def _normalize_lists(dataset: HFDataset) -> HFDataset:
    """List-like columns as ``large_list<element>``, so runs agree on their type.

    The same column can come in several list types: a fixed-size or a plain list, or
    Ray Data's tensor extension — a struct of the row's values (``data``) and shape —
    which Ray writes for columns that passed through numpy batches (e.g. the coverage
    grids of preprocessing.tiling_blocks.OverlayGrid). Concatenating runs that differ
    would otherwise fall back to casting the column to string (_concatenate).
    """
    table = _normalized(dataset.data)
    return dataset if table is dataset.data else HFDataset(table)


def _normalized(table: Any) -> Any:
    """_normalize_lists on a ``datasets`` table; the same table if nothing changes.

    A split cached in several files (e.g. the 32 x 32 grids of a large cohort) loads as
    a ConcatenationTable, which has no ``set_column``: its blocks, memory-mapped tables,
    are normalized one by one instead, so the data stays memory-mapped (and cheap to
    pickle to DataLoader workers).
    """
    if isinstance(table, ConcatenationTable):
        blocks = [[_normalized(block) for block in row] for row in table.blocks]
        unchanged = all(
            new is old
            for row, old_row in zip(blocks, table.blocks, strict=True)
            for new, old in zip(row, old_row, strict=True)
        )
        return table if unchanged else ConcatenationTable.from_blocks(blocks)
    for index, name in enumerate(table.column_names):
        column = table.column(name)
        kind = column.type.storage_type if isinstance(column.type, pa.ExtensionType) else column.type
        if isinstance(column.type, pa.ExtensionType):
            column = pa.chunked_array([chunk.storage for chunk in column.chunks], type=kind)
        if pa.types.is_struct(kind) and {kind.field(i).name for i in range(kind.num_fields)} == {"data", "shape"}:
            column = pc.struct_field(column, "data")
            kind = column.type
        if (pa.types.is_list(kind) or pa.types.is_fixed_size_list(kind)) and not pa.types.is_large_list(kind):
            column = column.cast(pa.large_list(kind.value_type))
        if column.type != table.column(name).type:
            table = table.set_column(index, name, column)
    return table


class MetaDataset[T](MetaTiledSlides[T]):
    """Adds MLflow-URI loading and lazy threshold filtering on top of ``MetaTiledSlides``.

    ``MetaTiledSlides`` (from ``rationai.mlkit``) already does the hard part: given a
    ``(slides, tiles)`` pair it builds a fast ``slide_id -> tile row indices`` lookup
    via Arrow's native ``group_by``, and turns ``generate_datasets()``'s per-slide tile
    views into one flat, globally tile-indexable ``torch.utils.data.ConcatDataset``.

    This class only has to produce that ``(slides, tiles)`` pair — loaded from MLflow
    artifact URIs, with caller-supplied column thresholds already applied — so any
    concrete subclass (e.g. ``EmbeddingClassificationDataset``) gets the same loading
    and filtering behaviour for free just by inheriting from this instead of
    ``MetaTiledSlides`` directly.

    A classmethod constructor (``from_uris``) is used instead of parameterizing the
    base class by the concrete dataset type, since a type parameter is not an actual
    class object and cannot appear in a base-class list at runtime — ``cls`` dispatch
    achieves the same "which concrete dataset gets built" behaviour validly.
    """

    def __init__(
        self,
        uris: Iterable[str],
        tile_filters: Sequence[FilterCondition] = (),
        slide_filters: Sequence[FilterCondition] = (),
    ) -> None:
        uris = list(uris)
        slides = self._load_table(uris, "slides").with_format("numpy")
        tiles = self._load_table(uris, "tiles").with_format("numpy")

        slides = self._select_slides(slides, tiles)
        slides, tiles = self._preprocess_dataset(slides, tiles, tile_filters, slide_filters)
        super().__init__(slides=slides, tiles=tiles)

    def _select_slides(self, slides: HFDataset, tiles: HFDataset) -> HFDataset:
        """Hook to drop whole slides based on their tiles; keeps every slide by default."""
        return slides

    @staticmethod
    def _load_table(uris: Sequence[str], table: str) -> HFDataset:
        """Load one table (``slides`` or ``tiles``) from every run and concatenate them."""
        return MetaDataset._concatenate(
            [_normalize_lists(load_dataset(uris=[str(Path(uri) / table)])["train"]) for uri in uris]
        )

    @staticmethod
    def _concatenate(parts: Sequence[HFDataset]) -> HFDataset:
        """``concatenate_datasets``, tolerating columns typed differently across runs.

        Runs from different cohorts can disagree on a column's inferred type — e.g.
        purely numeric case IDs read as int64 in one cohort, text IDs as strings in
        another — which ``concatenate_datasets`` refuses. Such columns are cast to
        string first. An all-null column (type ``null``) aligns with any type as is.
        """
        null = repr(Value("null"))
        types: dict[str, set[str]] = {}
        for part in parts:
            for name, feature in part.features.items():
                types.setdefault(name, set()).add(repr(feature))
        mismatched = [name for name, seen in types.items() if len(seen - {null}) > 1]

        aligned = []
        for part in parts:
            for name in mismatched:
                if name in part.features:
                    part = part.cast_column(name, Value("string"))
            aligned.append(part)
        return concatenate_datasets(aligned)

    @staticmethod
    def _preprocess_dataset(
        slides: HFDataset,
        tiles: HFDataset,
        tile_filters: Sequence[FilterCondition],
        slide_filters: Sequence[FilterCondition],
    ) -> tuple[HFDataset, HFDataset]:
        """Drop rows that fail any filter condition, before any indexing happens.

        Every condition is combined into one boolean mask via vectorized, whole-column
        comparisons (one O(n) numpy pass per condition — no Python callback per row),
        then applied via ``select()`` + ``flatten_indices()`` (see ``_select``) — one
        compaction pass over just the kept rows, not a per-row Python callback like
        ``filter()`` would use.
        """
        slides = MetaDataset._select(slides, slide_filters)

        # Tiles belonging to a slide dropped by slide_filters would otherwise sit in
        # tiles unreferenced by anything in generate_datasets() (which only iterates
        # self.slides) — harmless for correctness, but wasted memory and a larger
        # group_by in MetaTiledSlides._build_tile_index. Folding the membership check
        # into tile_filters keeps this a single combined mask + select() pass.
        surviving_slide_ids = set(column_values(slides, "id").tolist())
        orphan_filter = FilterCondition("slide_id", "in", surviving_slide_ids)
        tiles = MetaDataset._select(tiles, (*tile_filters, orphan_filter), slides=slides)

        return slides, tiles

    @staticmethod
    def _select(
        dataset: HFDataset, filters: Sequence[FilterCondition], slides: HFDataset | None = None
    ) -> HFDataset:
        if not filters:
            return dataset

        mask = np.ones(len(dataset), dtype=bool)
        for condition in filters:
            keep = condition.mask(dataset)
            if condition.cohort is not None:
                if slides is None or "cohort_id" not in slides.column_names:
                    raise ValueError(f"Filter on {condition.column!r} is scoped to cohort "
                                     f"{condition.cohort!r}, which needs slides with a cohort_id column")
                cohort_ids = column_values(slides, "id")[column_values(slides, "cohort_id") == condition.cohort]
                keep |= ~np.isin(column_values(dataset, "slide_id"), cohort_ids)
            mask &= keep

        selected = dataset.select(np.flatnonzero(mask))
        # select() is normally a pure indices-overlay (no data copy) — but
        # MetaTiledSlides._build_tile_index reads `tiles.data` directly, which is
        # the *unfiltered* base table and ignores that overlay entirely. Without
        # flattening, the row positions it computes are stale (sized to the
        # original table) and a later filter_tiles_by_slide() select() on the
        # already-filtered table indexes out of range. flatten_indices() bakes the
        # selection into a real, compacted base table — a one-time O(kept rows)
        # rewrite, done once here rather than being silently wrong downstream.
        # Flattened in Arrow format: flatten_indices() maps over the dataset's format,
        # and numpy can't hold list columns with null rows (e.g. a coverage grid of
        # slides without that mask).
        fmt = dataset.format
        return (
            selected.with_format("arrow")
            .flatten_indices()
            .with_format(fmt["type"], columns=fmt["columns"], output_all_columns=fmt["output_all_columns"], **fmt["format_kwargs"])
        )

from __future__ import annotations

import contextlib
import operator as _op
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from datasets import Dataset as HFDataset
from datasets import Value, concatenate_datasets
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
            [load_dataset(uris=[str(Path(uri) / table)])["train"] for uri in uris]
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
        return selected.flatten_indices()

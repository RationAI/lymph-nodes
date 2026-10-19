from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.trainer.states import TrainerFn

from lymph_nodes.data.datasets.meta_dataset import column_values


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lightning import LightningModule, Trainer

    from lymph_nodes.data.datasets.embedding import EmbeddingClassificationDataset
    from lymph_nodes.typedefs import Batch, Outputs


# The 8 neighbours of a tile on its slide's tile grid.
_NEIGHBOURS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


class LesionMetrics(Callback):
    """Lesion-level detection metrics of the final evaluations, logged to MLflow.

    Tile metrics count tiles, but a pathologist reads lesions: an ITC is one to a few
    tiles, a macrometastasis hundreds, and finding one tile of each is what matters.
    During ``validate`` and ``test`` (not the validation inside ``fit``), for the
    lymph node ``cohorts``:

    - a *lesion* is an 8-connected group of positive tiles (the tile labels: annotation
      coverage > 0) on one slide's tile grid; it is detected at a threshold when its
      best tile reaches it,
    - false positives are the tiles flagged on *negative* slides (the slides' ``tumor``
      status); a *region* is an 8-connected group of them.

    The annotations are rough outlines that may include lymphoid tissue, and single
    tumour cells may be left unannotated (or be mimicked by benign cells), whereas
    groups of cells must not be missed. Lesions are therefore also split by size: the
    number of CK-stained cells, estimated from the brown pixels of the lesion's tiles
    (``brown_column``, counted over each tile's stride x stride core so that overlapping
    tiles are not counted twice) divided by ``cell_pixels``, the brown pixels of one
    stained cell at the tiling level (~300-500 at 0.5 um/px). Lesions of at least
    ``min_cells`` cells are *clusters*, the target; smaller ones are reported apart
    (``single/``) and not required. False-positive regions are split the same way.

    Logged as ``<phase>/lesions/...`` (all cohorts pooled) and
    ``<phase>/lesions/<cohort>/...``:

    - ``n_lesions``; ``sensitivity`` at probability 0.5,
    - ``sensitivity_fp<b>`` for each ``budget`` b: at the threshold that flags at most b
      tiles per negative slide on average (FROC-like; thresholds from the negative
      slides of all cohorts, so a cohort without negative slides is still evaluated),
    - pooled, at 0.5: ``fp_tiles_per_negative_slide``, ``fp_regions_per_negative_slide``
      and ``clean_negative_slides`` (the share without a flagged tile); and
      ``fp_tiles_per_negative_slide_full_sensitivity``, at the highest threshold that
      still detects every lesion,
    - with sizes (``brown_column`` present): ``clusters/`` and ``single/`` versions of
      ``n_lesions`` and the sensitivities; ``sensitivity_fpr<r>`` (and
      ``clusters/sensitivity_fpr<r>``) for each ``region_budget`` r, at the lowest
      threshold that flags at most r regions per negative slide on average (FROC in
      regions, what a reader sees); and, at 0.5, ``fp_cluster_regions_per_negative_slide``
      and ``fp_small_regions_per_negative_slide``.

    A tile's score is its probability; for per-patch outputs (SegmentationMetaArch) its
    patches' maximum, or mean with ``tile_score="mean"``, as in the tile metrics. Tile
    labels come from the samples' meta ``label`` (TileSegmentationDataset) or else the
    batch targets (EmbeddingClassificationDataset). Positive tiles on positive slides
    outside lesions don't occur; other tiles of positive slides are not counted as
    false positives, as unannotated tumour may lie there.
    """

    def __init__(
        self,
        cohorts: Sequence[str] | None = None,
        budgets: Sequence[int] = (5, 10, 25, 50, 100, 250),
        region_budgets: Sequence[float] = (0.5, 1, 2, 4, 8),
        min_cells: float = 3,
        cell_pixels: float = 400,
        brown_column: str = "brownish_coverage",
    ) -> None:
        self.cohorts = None if cohorts is None else set(cohorts)
        self.budgets = list(budgets)
        self.region_budgets = list(region_budgets)
        self.min_cells = min_cells
        self.cell_pixels = cell_pixels
        self.brown_column = brown_column
        # Per batch: slide positions, tile_x, tile_y, tile scores, tile labels.
        self._tiles: list[tuple[np.ndarray, ...]] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._collect(pl_module, outputs, batch)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._log(pl_module, trainer.datamodule.val, "validation")  # type: ignore[attr-defined]

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect(pl_module, outputs, batch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log(pl_module, trainer.datamodule.test, "test")  # type: ignore[attr-defined]

    def _collect(
        self, pl_module: LightningModule, outputs: Outputs, batch: Batch
    ) -> None:
        _, targets, meta = batch
        probabilities = torch.sigmoid(outputs.detach().float())
        if probabilities.ndim > 1:  # per patch: (B, g, g)
            flat = probabilities.flatten(1)
            mean = getattr(pl_module, "tile_score", "max") == "mean"
            probabilities = flat.mean(dim=1) if mean else flat.amax(dim=1)
        labels = meta.get("label", targets)
        self._tiles.append(
            (
                meta["slide"].cpu().numpy(),
                meta["tile_x"].cpu().numpy(),
                meta["tile_y"].cpu().numpy(),
                probabilities.cpu().numpy(),
                (torch.as_tensor(labels).float() > 0.5).cpu().numpy(),
            )
        )

    def _log(
        self,
        pl_module: LightningModule,
        dataset: EmbeddingClassificationDataset,
        phase: str,
    ) -> None:
        if not self._tiles:
            return
        slide, tile_x, tile_y, score, label = (
            np.concatenate(c) for c in zip(*self._tiles, strict=True)
        )
        self._tiles.clear()

        slides = dataset.slides
        cohorts = column_values(slides, "cohort_id").astype(str)
        keep = (
            np.ones(len(slide), bool)
            if self.cohorts is None
            else np.isin(cohorts[slide], list(self.cohorts))
        )
        if not keep.any():
            return
        stride_x = column_values(slides, "stride_x")[slide[keep]]
        stride_y = column_values(slides, "stride_y")[slide[keep]]
        brown = self._brown(dataset, slide[keep], tile_x[keep], tile_y[keep])
        values = lesion_metrics(
            slide=slide[keep],
            grid_x=tile_x[keep] // stride_x,
            grid_y=tile_y[keep] // stride_y,
            score=score[keep],
            label=label[keep],
            slide_negative=~column_values(slides, "tumor").astype(bool),
            slide_cohort=cohorts,
            budgets=self.budgets,
            cells=None
            if brown is None
            else brown * stride_x * stride_y / self.cell_pixels,
            min_cells=self.min_cells,
            region_budgets=self.region_budgets,
        )
        pl_module.log_dict(
            {f"{phase}/lesions/{key}": value for key, value in values.items()}
        )

    def _brown(
        self,
        dataset: EmbeddingClassificationDataset,
        slide: np.ndarray,
        tile_x: np.ndarray,
        tile_y: np.ndarray,
    ) -> np.ndarray | None:
        """``brown_column`` of the given tiles, or None if the tiles have no such column."""
        if self.brown_column not in dataset.tiles.column_names:
            return None
        index = dataset._get_batch_index()
        brown = dataset.tiles.data.column(self.brown_column).to_numpy(
            zero_copy_only=False
        )
        brown = np.nan_to_num(brown[index.rows].astype(float))

        def key(s: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return (
                (s.astype(np.int64) << 42)
                | (x.astype(np.int64) << 21)
                | y.astype(np.int64)
            )

        table = key(index.slide_of, index.tile_x, index.tile_y)
        order = np.argsort(table)
        found = order[np.searchsorted(table[order], key(slide, tile_x, tile_y))]
        return brown[found]


def lesion_metrics(
    slide: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score: np.ndarray,
    label: np.ndarray,
    slide_negative: np.ndarray,
    slide_cohort: np.ndarray,
    budgets: Iterable[int],
    cells: np.ndarray | None = None,
    min_cells: float = 3,
    region_budgets: Iterable[float] = (),
) -> dict[str, float]:
    """The metrics of LesionMetrics from per-tile arrays (see there).

    ``slide`` indexes ``slide_negative`` and ``slide_cohort``; ``grid_x``, ``grid_y``
    are tile positions on the slide's tile grid (tile coordinate // stride). ``cells``,
    if given, is each tile's estimated number of CK-stained cells, for the size split.
    """
    tile_cells = np.zeros(len(slide)) if cells is None else np.asarray(cells, float)
    # Per lesion: cohort, best score, estimated cells.
    lesions: list[tuple[str, float, float]] = []
    negative_tiles: dict[int, np.ndarray] = {}  # negative slide -> its tile indices
    for position in np.unique(slide):
        tiles = slide == position
        if slide_negative[position]:
            negative_tiles[int(position)] = np.flatnonzero(tiles)
            continue
        positive = np.flatnonzero(tiles & label)
        grid = np.stack([grid_x[positive], grid_y[positive]], axis=1)
        for component in _components(grid):
            members = positive[component]
            lesions.append(
                (
                    str(slide_cohort[position]),
                    float(score[members].max()),
                    float(tile_cells[members].sum()),
                )
            )

    negatives = np.concatenate(
        [score[t] for t in negative_tiles.values()] or [np.empty(0)]
    )
    n_negative_slides = len(negative_tiles)
    thresholds = {"": 0.5}
    if n_negative_slides:
        # Highest-first; flagging score > the (k+1)-th highest flags at most k tiles.
        ranked = np.sort(negatives)[::-1]
        for budget in budgets:
            k = budget * n_negative_slides
            thresholds[f"_fp{budget}"] = (
                float(np.nextafter(ranked[k], np.inf)) if k < len(ranked) else -np.inf
            )
        if cells is not None:
            for region_budget, threshold in _region_thresholds(
                negative_tiles, grid_x, grid_y, score, ranked, region_budgets
            ).items():
                thresholds[f"_fpr{region_budget:g}"] = threshold

    values: dict[str, float] = {}
    lesion_cohorts = np.array([c for c, _, _ in lesions])
    lesion_scores = np.array([s for _, s, _ in lesions])
    lesion_cells = np.array([n for _, _, n in lesions])
    groups = [("", np.ones(len(lesions), bool))] + [
        (f"{cohort}/", lesion_cohorts == cohort) for cohort in np.unique(lesion_cohorts)
    ]
    sizes = [("", np.ones(len(lesions), bool))]
    if cells is not None:
        sizes += [
            ("clusters/", lesion_cells >= min_cells),
            ("single/", lesion_cells < min_cells),
        ]
    for prefix, members in groups:
        for size, in_size in sizes:
            selected = members & in_size
            values[f"{prefix}{size}n_lesions"] = float(selected.sum())
            if not selected.any():
                continue
            for suffix, threshold in thresholds.items():
                values[f"{prefix}{size}sensitivity{suffix}"] = float(
                    (lesion_scores[selected] >= threshold).mean()
                )

    if n_negative_slides:
        regions = [
            _flagged_regions(t, grid_x, grid_y, score, 0.5)
            for t in negative_tiles.values()
        ]
        values["fp_tiles_per_negative_slide"] = float(
            np.mean([(score[t] >= 0.5).sum() for t in negative_tiles.values()])
        )
        values["fp_regions_per_negative_slide"] = float(
            np.mean([len(r) for r in regions])
        )
        values["clean_negative_slides"] = float(np.mean([not r for r in regions]))
        if cells is not None:
            region_cells = [
                [tile_cells[members].sum() for members in r] for r in regions
            ]
            values["fp_cluster_regions_per_negative_slide"] = float(
                np.mean([sum(n >= min_cells for n in r) for r in region_cells])
            )
            values["fp_small_regions_per_negative_slide"] = float(
                np.mean([sum(n < min_cells for n in r) for r in region_cells])
            )
        if lesions:
            values["fp_tiles_per_negative_slide_full_sensitivity"] = float(
                (negatives >= lesion_scores.min()).sum() / n_negative_slides
            )
    return values


def _flagged_regions(
    tiles: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> list[np.ndarray]:
    """The regions (8-connected) of the ``tiles`` scoring at least ``threshold``, as tile indices."""
    flagged = tiles[score[tiles] >= threshold]
    grid = np.stack([grid_x[flagged], grid_y[flagged]], axis=1)
    return [flagged[component] for component in _components(grid)]


def _region_thresholds(
    negative_tiles: dict[int, np.ndarray],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score: np.ndarray,
    ranked: np.ndarray,
    budgets: Iterable[float],
) -> dict[float, float]:
    """Per budget r, the lowest threshold flagging at most r regions per negative slide.

    Candidate thresholds are negative tile scores at geometrically spaced ranks; the
    region count is not monotonic in the threshold (regions merge), so the lowest
    candidate within the budget whose higher candidates are all within it too is taken.
    """
    budgets = sorted(budgets)
    if not budgets or not len(ranked):
        return {}
    n_slides = len(negative_tiles)
    ranks = np.unique(np.geomspace(1, len(ranked), 120).astype(int))
    chosen = dict.fromkeys(budgets, np.inf)
    open_budgets = list(budgets)
    for rank in ranks:  # thresholds from high to low
        threshold = float(ranked[rank - 1])
        per_slide = (
            np.mean(
                [
                    len(_flagged_regions(t, grid_x, grid_y, score, threshold))
                    for t in negative_tiles.values()
                ]
            )
            if n_slides
            else 0.0
        )
        for budget in list(open_budgets):
            if per_slide <= budget:
                chosen[budget] = threshold
            else:
                open_budgets.remove(budget)
        if not open_budgets:
            break
    return chosen


def _components(cells: np.ndarray) -> list[np.ndarray]:
    """8-connected components of grid cells (N, 2), as index arrays into ``cells``."""
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    seen = np.zeros(len(cells), bool)
    components = []
    for start in range(len(cells)):
        if seen[start]:
            continue
        seen[start] = True
        queue, members = deque([start]), [start]
        while queue:
            x, y = cells[queue.popleft()]
            for dx, dy in _NEIGHBOURS:
                neighbour = index.get((int(x) + dx, int(y) + dy))
                if neighbour is not None and not seen[neighbour]:
                    seen[neighbour] = True
                    queue.append(neighbour)
                    members.append(neighbour)
        components.append(np.array(members))
    return components

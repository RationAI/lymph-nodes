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

    Logged as ``<phase>/lesions/...`` (all cohorts pooled) and
    ``<phase>/lesions/<cohort>/...``:

    - ``n_lesions``; ``sensitivity`` at probability 0.5,
    - ``sensitivity_fp<b>`` for each ``budget`` b: at the threshold that flags at most b
      tiles per negative slide on average (FROC-like; thresholds from the negative
      slides of all cohorts, so a cohort without negative slides is still evaluated),
    - pooled, at 0.5: ``fp_tiles_per_negative_slide``, ``fp_regions_per_negative_slide``
      and ``clean_negative_slides`` (the share without a flagged tile); and
      ``fp_tiles_per_negative_slide_full_sensitivity``, at the highest threshold that
      still detects every lesion.

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
    ) -> None:
        self.cohorts = None if cohorts is None else set(cohorts)
        self.budgets = list(budgets)
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
        values = lesion_metrics(
            slide=slide[keep],
            grid_x=tile_x[keep] // column_values(slides, "stride_x")[slide[keep]],
            grid_y=tile_y[keep] // column_values(slides, "stride_y")[slide[keep]],
            score=score[keep],
            label=label[keep],
            slide_negative=~column_values(slides, "tumor").astype(bool),
            slide_cohort=cohorts,
            budgets=self.budgets,
        )
        pl_module.log_dict(
            {f"{phase}/lesions/{key}": value for key, value in values.items()}
        )


def lesion_metrics(
    slide: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score: np.ndarray,
    label: np.ndarray,
    slide_negative: np.ndarray,
    slide_cohort: np.ndarray,
    budgets: Iterable[int],
) -> dict[str, float]:
    """The metrics of LesionMetrics from per-tile arrays (see there).

    ``slide`` indexes ``slide_negative`` and ``slide_cohort``; ``grid_x``, ``grid_y``
    are tile positions on the slide's tile grid (tile coordinate // stride).
    """
    lesions: list[tuple[str, float]] = []  # (cohort, best score) per lesion
    negative_scores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for position in np.unique(slide):
        tiles = slide == position
        if slide_negative[position]:
            negative_scores[int(position)] = (
                grid_x[tiles],
                grid_y[tiles],
                score[tiles],
            )
            continue
        positive = tiles & label
        cells = np.stack([grid_x[positive], grid_y[positive]], axis=1)
        for component in _components(cells):
            lesions.append(
                (str(slide_cohort[position]), float(score[positive][component].max()))
            )

    negatives = np.concatenate(
        [s for _, _, s in negative_scores.values()] or [np.empty(0)]
    )
    n_negative_slides = len(negative_scores)
    thresholds = {"": 0.5}
    if n_negative_slides:
        # Highest-first; flagging score > the (k+1)-th highest flags at most k tiles.
        ranked = np.sort(negatives)[::-1]
        for budget in budgets:
            k = budget * n_negative_slides
            thresholds[f"_fp{budget}"] = (
                float(np.nextafter(ranked[k], np.inf)) if k < len(ranked) else -np.inf
            )

    values: dict[str, float] = {}
    lesion_cohorts = np.array([c for c, _ in lesions])
    lesion_scores = np.array([s for _, s in lesions])
    for prefix, members in [("", np.ones(len(lesions), bool))] + [
        (f"{cohort}/", lesion_cohorts == cohort) for cohort in np.unique(lesion_cohorts)
    ]:
        values[f"{prefix}n_lesions"] = float(members.sum())
        if not members.any():
            continue
        for suffix, threshold in thresholds.items():
            values[f"{prefix}sensitivity{suffix}"] = float(
                (lesion_scores[members] >= threshold).mean()
            )

    if n_negative_slides:
        flagged = [(s >= 0.5) for _, _, s in negative_scores.values()]
        values["fp_tiles_per_negative_slide"] = float(
            np.mean([f.sum() for f in flagged])
        )
        values["fp_regions_per_negative_slide"] = float(
            np.mean(
                [
                    len(_components(np.stack([x[f], y[f]], axis=1)))
                    for (x, y, _), f in zip(
                        negative_scores.values(), flagged, strict=True
                    )
                ]
            )
        )
        values["clean_negative_slides"] = float(np.mean([not f.any() for f in flagged]))
        if lesions:
            values["fp_tiles_per_negative_slide_full_sensitivity"] = float(
                (negatives >= lesion_scores.min()).sum() / n_negative_slides
            )
    return values


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

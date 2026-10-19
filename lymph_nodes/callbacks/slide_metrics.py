from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.trainer.states import TrainerFn
from torchmetrics.functional.classification import binary_auroc

from lymph_nodes.data.datasets.meta_dataset import column_values


if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning import LightningModule, Trainer

    from lymph_nodes.data.datasets.embedding import EmbeddingClassificationDataset
    from lymph_nodes.typedefs import Batch, Outputs


class SlideMetrics(Callback):
    """Slide-level metrics of the final evaluations, logged to MLflow.

    Most positive lymph node slides have no annotation, only their status (the slides'
    ``tumor``): tile metrics can't use them, but whether a slide is flagged at all can.
    During ``validate`` and ``test`` (not the validation inside ``fit``), for the slides
    of ``cohorts`` (all with a tumour status when None; TMA slides have none):

    - a slide's score is its highest tile probability (``max``), and the mean of its
      ``top_k`` highest (``top<k>``), less sensitive to a single artefact tile; a tile's
      probability is, for per-patch outputs (SegmentationMetaArch), its patches'
      maximum, or mean with ``tile_score="mean"``, as in the tile metrics,
    - every evaluated slide counts, the ``ignore``-flagged ones included (unannotated
      positive slides, EmbeddingClassificationDataset's
      ``ignore_unannotated_positive_slides``); a slide without evaluated tiles doesn't.

    Logged as ``<phase>/slides/<score>/...`` (all cohorts pooled) and
    ``<phase>/slides/<score>/<cohort>/...``: ``n_slides``, ``n_positive``; ``auroc``;
    ``sensitivity`` and ``specificity`` at probability 0.5; ``specificity_full_
    sensitivity``, the share of negative slides below the lowest positive one (how many
    negative slides a reader could skip without missing a positive one); and
    ``sensitivity_spec<s>`` for each of ``specificities``, at the lowest threshold that
    leaves at least that share of negative slides unflagged.
    """

    def __init__(
        self,
        cohorts: Sequence[str] | None = None,
        top_k: int = 5,
        specificities: Sequence[float] = (0.5, 0.75, 0.9),
    ) -> None:
        self.cohorts = None if cohorts is None else set(cohorts)
        self.top_k = top_k
        self.specificities = list(specificities)
        # Per batch: slide positions and tile scores.
        self._tiles: list[tuple[np.ndarray, np.ndarray]] = []

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
        probabilities = torch.sigmoid(outputs.detach().float())
        if probabilities.ndim > 1:  # per patch: (B, g, g)
            flat = probabilities.flatten(1)
            mean = getattr(pl_module, "tile_score", "max") == "mean"
            probabilities = flat.mean(dim=1) if mean else flat.amax(dim=1)
        self._tiles.append(
            (batch[2]["slide"].cpu().numpy(), probabilities.cpu().numpy())
        )

    def _log(
        self,
        pl_module: LightningModule,
        dataset: EmbeddingClassificationDataset,
        phase: str,
    ) -> None:
        if not self._tiles:
            return
        slide, score = (np.concatenate(c) for c in zip(*self._tiles, strict=True))
        self._tiles.clear()

        slides = dataset.slides
        tumor = column_values(slides, "tumor")
        cohorts = column_values(slides, "cohort_id").astype(str)
        evaluated = np.unique(slide)
        known = np.array(
            [tumor[s] is not None and tumor[s] == tumor[s] for s in evaluated]
        )
        evaluated = evaluated[known]  # TMA slides have no status (None / NaN)
        if self.cohorts is not None:
            evaluated = evaluated[np.isin(cohorts[evaluated], list(self.cohorts))]
        if not len(evaluated):
            return
        label = tumor[evaluated].astype(bool)

        values: dict[str, float] = {}
        for name, slide_score in self._slide_scores(slide, score, evaluated).items():
            values |= _slide_metrics(f"{name}", slide_score, label, self.specificities)
            for cohort in np.unique(cohorts[evaluated]):
                in_cohort = cohorts[evaluated] == cohort
                values |= _slide_metrics(
                    f"{name}/{cohort}",
                    slide_score[in_cohort],
                    label[in_cohort],
                    self.specificities,
                )
        pl_module.log_dict(
            {f"{phase}/slides/{key}": value for key, value in values.items()}
        )

    def _slide_scores(
        self, slide: np.ndarray, score: np.ndarray, evaluated: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Scores of the ``evaluated`` slides (positions), from their tiles' scores."""
        order = np.lexsort((-score, slide))  # by slide, best tile first
        slide, score = slide[order], score[order]
        starts = np.searchsorted(slide, evaluated)
        ends = np.searchsorted(slide, evaluated, side="right")
        top = [
            score[a : min(b, a + self.top_k)] for a, b in zip(starts, ends, strict=True)
        ]
        return {
            "max": score[starts],
            f"top{self.top_k}": np.array([t.mean() for t in top]),
        }


def _slide_metrics(
    prefix: str, score: np.ndarray, label: np.ndarray, specificities: Sequence[float]
) -> dict[str, float]:
    positive, negative = score[label], score[~label]
    values = {
        f"{prefix}/n_slides": float(len(label)),
        f"{prefix}/n_positive": float(label.sum()),
    }
    if len(positive):
        values[f"{prefix}/sensitivity"] = float((positive >= 0.5).mean())
    if len(negative):
        values[f"{prefix}/specificity"] = float((negative < 0.5).mean())
    if len(positive) and len(negative):
        values[f"{prefix}/auroc"] = float(
            binary_auroc(torch.from_numpy(score), torch.from_numpy(label).int())
        )
        values[f"{prefix}/specificity_full_sensitivity"] = float(
            (negative < positive.min()).mean()
        )
        ranked = np.sort(negative)
        for specificity in specificities:
            # The negative slide at that quantile is the highest one left unflagged.
            threshold = ranked[int(np.ceil(specificity * len(ranked))) - 1]
            values[f"{prefix}/sensitivity_spec{specificity:g}"] = float(
                (positive > threshold).mean()
            )
    return values

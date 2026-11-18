from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.trainer.states import TrainerFn

from lymph_nodes.callbacks.lesion_metrics import _region_thresholds
from lymph_nodes.data.datasets.meta_dataset import column_values
from lymph_nodes.segmentation_meta_arch import tile_scores


if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning import LightningModule, Trainer

    from lymph_nodes.data.augmentations import CellPasting
    from lymph_nodes.data.datasets.segmentation import TileSegmentationDataset


class SyntheticLesionMetrics(Callback):
    """Detection of synthetic isolated tumour cells in the validation during fit.

    A selection signal without positive lymph nodes. The validation during fit holds
    none (no lymph node annotation may steer the training), and its TMA patches are
    near-perfect from the first epoch on, so nothing in it tells whether isolated tumour
    cells are found; its negative lymph node slides alone would favour a model that
    flags nothing. Here a fixed set of ``count`` validation tiles (the same every epoch
    and every run, by ``seed``) gets cells pasted by ``cell_pasting``
    (TileSegmentationDataset.synthetic_lesions): the held-out fold's TMA tumour cells in
    its negative lymph node tiles, inside the node. Each epoch they pass the model (its
    evaluating head, under the trainer's precision); a pasted tile's score is the
    highest probability over its patches with pasted cells.

    The thresholds come from the negative lymph node slides of the validation (of
    ``cohorts``), as LesionMetrics' region budgets: for each ``r`` of
    ``region_budgets``, the lowest threshold that flags at most ``r`` regions
    (8-connected flagged tiles) per negative slide. Logged as ``validation/synthetic/``:

    - ``sensitivity_fpr<r>``: the share of the pasted tiles found at that threshold,
    - ``froc``: their mean — as CAMELYON16's FROC score averages the sensitivities at
      several false-positive rates — the monitor of the scenarios' checkpoint and
      early stopping,
    - ``sensitivity`` at probability 0.5 and ``n_lesions``.

    Pasted cells are no lymph node metastases: though from held-out cores and slides,
    they are the training's own kind of positive. The measure picks a run's epoch; the
    lymph node evaluation compares runs.
    """

    def __init__(
        self,
        cell_pasting: CellPasting,
        count: int = 1000,
        seed: int = 0,
        cohorts: Sequence[str] = ("mmci_ihc_2023",),
        region_budgets: Sequence[float] = (0.5, 1, 2, 4, 8),
        batch_size: int = 64,
    ) -> None:
        self.cell_pasting = cell_pasting
        self.count = count
        self.seed = seed
        self.cohorts = list(cohorts)
        self.region_budgets = sorted(float(r) for r in region_budgets)
        self.batch_size = batch_size
        # Per validation batch: slide positions, tile_x, tile_y, tile scores.
        self._tiles: list[tuple[np.ndarray, ...]] = []
        # The pasted tiles and their target grids, built at the first validation.
        self._synthetic: tuple[np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _active(trainer: Trainer) -> bool:
        return trainer.state.fn == TrainerFn.FITTING and not trainer.sanity_checking

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._active(trainer):
            return
        meta = batch[2]
        probabilities = torch.sigmoid(outputs.detach().float())
        if probabilities.ndim > 1:  # per patch: the tile scores of the tile metrics
            probabilities = tile_scores(
                probabilities,
                meta,
                getattr(pl_module, "tile_score", "max"),
                getattr(pl_module, "tile_region", "full"),
            )
        self._tiles.append(
            (
                meta["slide"].cpu().numpy(),
                meta["tile_x"].cpu().numpy(),
                meta["tile_y"].cpu().numpy(),
                probabilities.cpu().numpy(),
            )
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self._active(trainer) or not self._tiles:
            self._tiles.clear()
            return
        dataset: TileSegmentationDataset = trainer.datamodule.val  # type: ignore[attr-defined]
        slide, tile_x, tile_y, score = (
            np.concatenate(c) for c in zip(*self._tiles, strict=True)
        )
        self._tiles.clear()
        slides = dataset.slides
        negative = ~column_values(slides, "tumor").astype(bool)
        cohorts = column_values(slides, "cohort_id").astype(str)
        keep = negative[slide] & np.isin(cohorts[slide], self.cohorts)
        if not keep.any():
            return
        slide, score = slide[keep], score[keep]
        grid_x = tile_x[keep] // column_values(slides, "stride_x")[slide]
        grid_y = tile_y[keep] // column_values(slides, "stride_y")[slide]
        negative_tiles = {int(s): np.flatnonzero(slide == s) for s in np.unique(slide)}
        thresholds = _region_thresholds(
            negative_tiles,
            grid_x,
            grid_y,
            score,
            np.sort(score)[::-1],
            self.region_budgets,
        )
        lesions = self._lesion_scores(trainer, pl_module, dataset)
        values = {
            "n_lesions": float(len(lesions)),
            "sensitivity": float((lesions >= 0.5).mean()),
        }
        for budget in self.region_budgets:
            values[f"sensitivity_fpr{budget:g}"] = float(
                (lesions >= thresholds.get(budget, np.inf)).mean()
            )
        values["froc"] = float(
            np.mean([values[f"sensitivity_fpr{b:g}"] for b in self.region_budgets])
        )
        pl_module.log_dict(
            {f"validation/synthetic/{key}": value for key, value in values.items()}
        )

    def _lesion_scores(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        dataset: TileSegmentationDataset,
    ) -> np.ndarray:
        """Per pasted tile, the highest probability over its patches with pasted cells."""
        if self._synthetic is None:
            self._synthetic = dataset.synthetic_lesions(
                self.cell_pasting, self.count, self.seed
            )
            print(f"synthetic lesions: {len(self._synthetic[0])} pasted tiles")
        tiles, grids = self._synthetic
        was_training = pl_module.training
        pl_module.eval()
        scores = []
        try:
            with torch.no_grad(), trainer.precision_plugin.forward_context():
                for start in range(0, len(tiles), self.batch_size):
                    batch = torch.from_numpy(tiles[start : start + self.batch_size])
                    targets = torch.from_numpy(grids[start : start + self.batch_size])
                    batch = batch.to(pl_module.device)
                    targets = targets.to(pl_module.device)
                    logits = pl_module._logits(batch, targets)  # type: ignore[operator]
                    probabilities = torch.sigmoid(logits.float())
                    pasted = torch.where(
                        targets > 0, probabilities, torch.zeros_like(probabilities)
                    )
                    scores.append(pasted.flatten(1).amax(1).cpu().numpy())
        finally:
            pl_module.train(was_training)
        return np.concatenate(scores)

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
    from collections.abc import Mapping, Sequence

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
      ``froc`` their mean (as CAMELYON16's FROC score averages the sensitivities at
      several false-positive rates), ``sensitivity`` at probability 0.5, ``n_lesions``;
    - for each of ``extra_lesions`` (name: CellPasting), the same under ``<name>/``:
      other kinds of pasted cells, reported only;
    - for each of ``decoys`` (name: CellPasting, e.g. ``cut="unstained"``: cells with
      no stain from the same donors), ``decoys/<name>/rate_fpr<r>`` and ``rate`` (at
      0.5): the share of decoy tiles flagged at their pastes, and ``baseline_fpr<r>``
      and ``baseline``: the same regions of the tiles before the pasting. A decoy rate
      above its baseline is the pasting itself flagged — a model flagging the paste,
      not the cell;
    - ``net_froc``: the mean over the budgets of the sensitivity less the highest decoy
      rate (``froc`` without decoys) — the monitor of the scenarios' checkpoint and
      early stopping, so that a model leaning on the pasting is not the one kept.

    Pasted cells are no lymph node metastases: though from held-out cores and slides,
    they are the training's own kind of positive. The measure picks a run's epoch; the
    lymph node evaluation compares runs. ``during_validate`` measures in ``validate``
    too (with DataModule's ``validate_on: val``, diagnostics of a checkpoint), else only
    in the validation during fit.
    """

    def __init__(
        self,
        cell_pasting: CellPasting,
        count: int = 1000,
        seed: int = 0,
        cohorts: Sequence[str] = ("mmci_ihc_2023",),
        region_budgets: Sequence[float] = (0.5, 1, 2, 4, 8),
        batch_size: int = 64,
        extra_lesions: Mapping[str, CellPasting] | None = None,
        decoys: Mapping[str, CellPasting] | None = None,
        during_validate: bool = False,
    ) -> None:
        self.cell_pasting = cell_pasting
        self.count = count
        self.seed = seed
        self.cohorts = list(cohorts)
        self.region_budgets = sorted(float(r) for r in region_budgets)
        self.batch_size = batch_size
        self.extra_lesions = dict(extra_lesions or {})
        self.decoys = dict(decoys or {})
        self.during_validate = during_validate
        # Per validation batch: slide positions, tile_x, tile_y, tile scores.
        self._tiles: list[tuple[np.ndarray, ...]] = []
        # Per set: the pasted tiles, their grids and the originals (decoys), built at
        # the first validation.
        self._sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}

    def _active(self, trainer: Trainer) -> bool:
        if trainer.state.fn == TrainerFn.VALIDATING:
            return self.during_validate
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
        thresholds = self._thresholds(dataset)
        self._tiles.clear()
        if thresholds is None:
            return
        self._build(dataset)
        budgets = [
            (f"_fpr{b:g}", thresholds.get(b, np.inf)) for b in self.region_budgets
        ]
        cuts = [("", 0.5), *budgets]
        values: dict[str, float] = {}
        sensitivity: dict[str, list[float]] = {}
        for name in ["", *self.extra_lesions]:
            scores, _ = self._scores(trainer, pl_module, name)
            prefix = f"{name}/" if name else ""
            values[f"{prefix}n_lesions"] = float(len(scores))
            for suffix, threshold in cuts:
                values[f"{prefix}sensitivity{suffix}"] = float(
                    (scores >= threshold).mean()
                )
            sensitivity[name] = [
                values[f"{prefix}sensitivity{suffix}"] for suffix, _ in budgets
            ]
            values[f"{prefix}froc"] = float(np.mean(sensitivity[name]))
        rates = []
        for name in self.decoys:
            scores, baseline = self._scores(trainer, pl_module, f"decoys/{name}")
            for suffix, threshold in cuts:
                values[f"decoys/{name}/rate{suffix}"] = float(
                    (scores >= threshold).mean()
                )
                if baseline is not None:
                    values[f"decoys/{name}/baseline{suffix}"] = float(
                        (baseline >= threshold).mean()
                    )
            rates.append(
                [values[f"decoys/{name}/rate{suffix}"] for suffix, _ in budgets]
            )
        worst = np.max(rates, axis=0) if rates else np.zeros(len(budgets))
        values["net_froc"] = float(np.mean(np.asarray(sensitivity[""]) - worst))
        pl_module.log_dict(
            {f"validation/synthetic/{key}": value for key, value in values.items()}
        )

    def _thresholds(
        self, dataset: TileSegmentationDataset
    ) -> dict[float, float] | None:
        """Per region budget, the lowest threshold within it on the negative slides."""
        slide, tile_x, tile_y, score = (
            np.concatenate(c) for c in zip(*self._tiles, strict=True)
        )
        slides = dataset.slides
        negative = ~column_values(slides, "tumor").astype(bool)
        cohorts = column_values(slides, "cohort_id").astype(str)
        keep = negative[slide] & np.isin(cohorts[slide], self.cohorts)
        if not keep.any():
            return None
        slide, score = slide[keep], score[keep]
        grid_x = tile_x[keep] // column_values(slides, "stride_x")[slide]
        grid_y = tile_y[keep] // column_values(slides, "stride_y")[slide]
        negative_tiles = {int(s): np.flatnonzero(slide == s) for s in np.unique(slide)}
        return _region_thresholds(
            negative_tiles,
            grid_x,
            grid_y,
            score,
            np.sort(score)[::-1],
            self.region_budgets,
        )

    def _build(self, dataset: TileSegmentationDataset) -> None:
        """The pasted sets, once: the lesions, the extra lesions, the decoys."""
        if self._sets:
            return
        pastings = {
            "": (self.cell_pasting, False),
            **{name: (p, False) for name, p in self.extra_lesions.items()},
            **{f"decoys/{name}": (p, True) for name, p in self.decoys.items()},
        }
        for name, (pasting, originals) in pastings.items():
            self._sets[name] = dataset.synthetic_lesions(
                pasting, self.count, self.seed, keep_originals=originals
            )
            print(
                f"synthetic lesions{f' {name}' if name else ''}: "
                f"{len(self._sets[name][0])} pasted tiles"
            )

    def _scores(
        self, trainer: Trainer, pl_module: LightningModule, name: str
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Per pasted tile of set ``name``, its highest probability over the pastes.

        And the same patches' highest probability in the originals, if kept.
        """
        tiles, grids, originals = self._sets[name]
        pasted = self._max_over(trainer, pl_module, tiles, grids)
        baseline = (
            None
            if originals is None
            else self._max_over(trainer, pl_module, originals, grids)
        )
        return pasted, baseline

    def _max_over(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        tiles: np.ndarray,
        grids: np.ndarray,
    ) -> np.ndarray:
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
                    where = torch.where(
                        targets > 0, probabilities, torch.zeros_like(probabilities)
                    )
                    scores.append(where.flatten(1).amax(1).cpu().numpy())
        finally:
            pl_module.train(was_training)
        return np.concatenate(scores)

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.trainer.states import TrainerFn
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder

from lymph_nodes.data.datasets.meta_dataset import column_values


if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from rationai.mlkit.lightning.loggers import MLFlowLogger

    from lymph_nodes.data.datasets.embedding import EmbeddingClassificationDataset
    from lymph_nodes.typedefs import Batch, Outputs


# Slide columns (written by preprocessing/tiling.py) describing the tiling level's grid.
_GEOMETRY = ("extent_x", "extent_y", "stride_x", "stride_y", "mpp_x", "mpp_y")


class PredictionMasks(Callback):
    """A prediction mask of every slide of the final evaluations, logged to MLflow.

    During ``validate`` and ``test`` (not the validation inside ``fit``), each evaluated
    slide gets a BigTIFF of the predicted cancer probability scaled to 0-255 (0 also
    where no tile was evaluated), logged as ``prediction_masks/<cohort>/<slide_name>.tiff``.

    The mask has one pixel per tile: the stride x stride cell at the tile's grid
    position, so its mpp is the tiling level's times the stride (xOpat scales it onto
    the slide). Tiles overlap (stride < tile extent), and the cells of the grid cover
    the slide exactly once. A cell starts at its tile's corner rather than being
    centered on the tile, which shifts the mask by (tile extent - stride) / 2 — 51 px of
    the tiling level for the 224/122 lymph node grid, under half a mask pixel. Scaling
    the mask up to shift it exactly would cost a full-resolution image per slide, and a
    MaskBuilder over the full overlapping tiles would have to work at a resolution of
    gcd(tile extent, stride): 2 px for that grid.

    Needs the steps to return their logits and the samples' meta to carry ``slide``
    (position in the dataset's ``slides``), ``tile_x`` and ``tile_y``.
    """

    def __init__(self) -> None:
        # Per batch: slide positions, tile_x, tile_y, probabilities.
        self._predictions: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []

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
            self._collect(outputs, batch)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._log_masks(trainer, trainer.datamodule.val)  # type: ignore[attr-defined]

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect(outputs, batch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_masks(trainer, trainer.datamodule.test)  # type: ignore[attr-defined]

    def _collect(self, outputs: Outputs, batch: Batch) -> None:
        _, _, meta = batch
        self._predictions.append(
            (
                meta["slide"].cpu().numpy(),
                meta["tile_x"].cpu().numpy(),
                meta["tile_y"].cpu().numpy(),
                torch.sigmoid(outputs.detach().float()).cpu().numpy(),
            )
        )

    def _log_masks(
        self, trainer: Trainer, dataset: EmbeddingClassificationDataset
    ) -> None:
        import pyvips

        if not self._predictions:
            return
        slide, tile_x, tile_y, probability = (
            np.concatenate(c) for c in zip(*self._predictions, strict=True)
        )
        self._predictions.clear()

        slides = dataset.slides
        names = column_values(slides, "slide_name")
        cohorts = (
            column_values(slides, "cohort_id")
            if "cohort_id" in slides.column_names
            else None
        )
        geometry = {column: column_values(slides, column) for column in _GEOMETRY}

        # The tiles of each slide, as one contiguous run of the sorted predictions.
        order = np.argsort(slide, kind="stable")
        positions, starts = np.unique(slide[order], return_index=True)
        with TemporaryDirectory() as tmp:
            for position, tiles in zip(
                positions, np.split(order, starts[1:]), strict=True
            ):
                directory = Path(
                    tmp, str(cohorts[position]) if cohorts is not None else "unknown"
                )
                directory.mkdir(exist_ok=True)
                slide_geometry = {
                    column: values[position] for column, values in geometry.items()
                }
                write_big_tiff(
                    pyvips.Image.new_from_array(
                        _prediction_mask(
                            slide_geometry,
                            tile_x[tiles],
                            tile_y[tiles],
                            probability[tiles],
                        )
                    ),
                    path=directory / f"{names[position]}.tiff",
                    mpp_x=float(slide_geometry["mpp_x"] * slide_geometry["stride_x"]),
                    mpp_y=float(slide_geometry["mpp_y"] * slide_geometry["stride_y"]),
                )
            cast("MLFlowLogger", trainer.logger).log_artifacts(tmp, "prediction_masks")


def _prediction_mask(
    slide: dict[str, Any],
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    """The uint8 probability mask of one slide, one pixel per tile (see PredictionMasks)."""
    stride = (int(slide["stride_y"]), int(slide["stride_x"]))

    # A "tile" of stride x stride source pixels per mask pixel.
    builder = MaskBuilder(
        source_extents=(int(slide["extent_y"]), int(slide["extent_x"])),
        source_tile_extent=stride,
        output_tile_extent=1,
        stride=stride,
    )
    try:
        builder.update_batch(
            probability[:, np.newaxis],
            np.stack([tile_y, tile_x], axis=1).astype(np.int64),
        )
        return np.round(builder.finalize()["mask"][0] * 255).astype(np.uint8)
    finally:
        builder.cleanup()

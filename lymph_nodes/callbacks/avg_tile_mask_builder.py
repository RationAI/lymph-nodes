import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import lightning.pytorch as pl
import mlflow
import numpy as np
import pandas as pd
import pyvips
import torch
from rationai.masks import write_big_tiff
from rationai.masks.mask_builders.mask_builder import MaskBuilder
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from lymph_nodes.typing import PredictSample, Sample
from prepro.utils import get_relative_dir_path


class TileMaskBuilder(MaskBuilder):
    def __init__(
        self,
        save_dir: Path | str,
        filename: str,
        extent_x: int,
        extent_y: int,
        mpp_x: float,
        mpp_y: float,
        tile_extent_x: int,
        tile_extent_y: int,
    ) -> None:
        super().__init__(save_dir, filename, extent_x, extent_y, mpp_x, mpp_y)

        self.kernel = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, tile_extent_y // 2),
                    np.linspace(1, 0, tile_extent_y // 2),
                ]
            ),
            np.concatenate(
                [
                    np.linspace(0, 1, tile_extent_x // 2),
                    np.linspace(1, 0, tile_extent_x // 2),
                ]
            ),
        )

        self.image = np.memmap(
            str(self.filename) + "_mask.nmp",
            dtype=np.float32,
            mode="w+",
            shape=(self.extent_y, self.extent_x),
        )

    def update(self, data: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> None:
        tiles = data.detach().cpu().numpy()
        xs_np = xs.detach().cpu().numpy()
        ys_np = ys.detach().cpu().numpy()

        for tile, x, y in zip(tiles, xs_np, ys_np, strict=True):
            mm_y, mm_x = self.image[y : y + tile.shape[0], x : x + tile.shape[1]].shape
            self.image[y : y + mm_y, x : x + mm_x] = (
                self.kernel[:mm_y, :mm_x] * tile[:mm_y, :mm_x]
            )

        self.image.flush()

    def save(self) -> Path:
        image_vips = pyvips.Image.new_from_array(self.image)

        image_vips *= 255
        image_vips = image_vips.cast(pyvips.BandFormat.UCHAR)

        path = self.filename.with_suffix(".tiff")
        write_big_tiff(image_vips, path, self.mpp_x, self.mpp_y)
        return path


if TYPE_CHECKING:
    from lymph_nodes.data.data_module import DataModule


class AvgTileMaskBuilder(MultiloaderLifecycle):
    def __init__(self) -> None:
        super().__init__()

        # Create temporary directories for output
        self.tmp_dir = tempfile.TemporaryDirectory()

    def __del__(self) -> None:
        # Remove the temp dir
        self.tmp_dir.cleanup()

    def on_test_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        return self.on_predict_dataloader_start(trainer, pl_module, dataloader_idx)

    def on_test_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        return self.on_predict_dataloader_end(trainer, pl_module, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        inputs, targets, metadata = batch

        return self.on_predict_batch_end(
            trainer, pl_module, outputs, (inputs, metadata), batch_idx, dataloader_idx
        )

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("DataModule", trainer.datamodule)
        self.slide = cast("pd.Series", datamodule.test.slides.iloc[dataloader_idx])

        # Initialize the mask builders for each output
        self.mask_builder = TileMaskBuilder(
            save_dir=self.tmp_dir.name,
            filename=Path(self.slide.path).stem,
            extent_x=self.slide.extent_x,
            extent_y=self.slide.extent_y,
            mpp_x=self.slide.mpp_x,
            mpp_y=self.slide.mpp_y,
            tile_extent_x=self.slide.tile_extent_x,
            tile_extent_y=self.slide.tile_extent_y,
        )

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self.mask_builder.save()

        # Log the heatmap to MLFlow
        pred_path = self.mask_builder.filename.with_suffix(".tiff")

        mlflow.log_artifact(
            str(pred_path),
            artifact_path=f"segmemtation_masks/{get_relative_dir_path(Path(self.slide.path))}",
        )

        pred_path.unlink()

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: PredictSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata = batch

        self.mask_builder.update(
            outputs,
            torch.tensor(metadata["x"]) + 64,
            torch.tensor(metadata["y"]) + 64,
        )

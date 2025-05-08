import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
from rationai.masks.mask_builders import TileMaskBuilder
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from lymph_nodes.typing import Outputs, PredictSample, SegSample
from prepro.utils import get_relative_dir_path


if TYPE_CHECKING:
    from lymph_nodes.data.data_module import DataModule


class TileMaskBuilder(MultiloaderLifecycle):
    def __init__(
        self, artifact_path: str, tile_builder_cstr: type[TileMaskBuilder]
    ) -> None:
        super().__init__()
        self.artifact_path = artifact_path
        self.tile_builder_cstr = tile_builder_cstr

        # Create temporary directories for output
        self.tmp_dir = tempfile.TemporaryDirectory()

    def __del__(self) -> None:
        # Remove the temp dir
        self.tmp_dir.cleanup()

    def init_builder(self) -> None:
        # Initialize the mask builders for each output
        self.mask_builder = self.tile_builder_cstr(
            save_dir=self.tmp_dir.name,
            filename=Path(self.slide.path).stem,
            extent_x=self.slide.extent_x,
            extent_y=self.slide.extent_y,
            mpp_x=self.slide.mpp_x,
            mpp_y=self.slide.mpp_y,
        )

    def on_test_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("DataModule", trainer.datamodule)
        self.slide = cast("pd.Series", datamodule.test.slides.iloc[dataloader_idx])

        self.init_builder()

    def on_test_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        return self.on_predict_dataloader_end(trainer, pl_module, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Outputs,
        batch: SegSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        inputs, mask_targets, cls_targets, metadata = batch

        return self.on_predict_batch_end(
            trainer, pl_module, outputs, (inputs, metadata), batch_idx, dataloader_idx
        )

    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast("DataModule", trainer.datamodule)
        self.slide = cast("pd.Series", datamodule.predict.slides.iloc[dataloader_idx])

        self.init_builder()

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self.mask_builder.save()

        # Log the heatmap to MLFlow
        pred_path = self.mask_builder.filename.with_suffix(".tiff")

        mlflow.log_artifact(
            str(pred_path),
            artifact_path=f"{self.artifact_path}/{get_relative_dir_path(Path(self.slide.path))}",
        )

        pred_path.unlink()

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Outputs,
        batch: PredictSample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata = batch

        self.mask_builder.update(
            outputs.masks,
            metadata["x"].detach()
            + round(self.slide.tile_extent_x * 0.1),  # assuming 0.75 overlap
            metadata["y"].detach()
            + round(self.slide.tile_extent_y * 0.1),  # assuming 0.75 overlap
        )

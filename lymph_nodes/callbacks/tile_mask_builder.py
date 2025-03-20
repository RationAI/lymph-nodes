import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
import torch
from rationai.masks.mask_builders import TileMaskBuilder
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle

from lymph_nodes.typing import PredictSample
from prepro.utils import get_relative_dir_path


if TYPE_CHECKING:
    from lymph_nodes.data.data_module import DataModule


class TileMaskBuilderTest(MultiloaderLifecycle):
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

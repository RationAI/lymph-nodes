from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import mlflow
import pandas as pd
from rationai.masks.mask_builders import ScalarMaskBuilder, TileMaskBuilder
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle


class TileMaskBuilderTest(MultiloaderLifecycle):
    def on_predict_dataloader_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer should have datamodule attribute")

        datamodule = cast(DataModule, trainer.datamodule)
        self.slide = cast(pd.Series, datamodule.predict.slides.iloc[dataloader_idx])

        # Create temporary directories for each output
        tmp_dir = Path(f"tmp_pred_dir")

        # Initialize the mask builders for each output
        self.mask_builder = TileMaskBuilder(
            save_dir=tmp_dir,
            filename=Path(self.slide.path).stem,
            extent_x=sefl.slide.extent_x,
            extent_y=self.slide.extent_y,
            mpp_X=self.slide.mpp_x,
            mpp_y=self.slide.mpp_y,
        )

    def on_predict_dataloader_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, dataloader_idx: int
    ) -> None:
        self.mask_builder.save()

        # Log the heatmap to MLFlow
        pred_path = self.mask_builder.filename.with_suffix(".tiff")
        mlflow.log_artifact(
            str(pred_path), artifact_path=f"predictions/{pred_path.name}"
        )

        pred_path.unlink()

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Outputs,
        batch: PredictInput,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata = batch

        # TileMaskBuilder needs whole tiles in update
        # So we set each pixel of a tile to the same value
        b, h, w = (
            outputs.shape[0],
            self.curr_slide_tile_extent,
            self.curr_slide_tile_extent,
        )
        # Output is (B, 3)
        tiles = outputs.view(b, 1, 1, 3).expand(b, h, w, 3)  # (B, H, W, 3)

        for i, mask_builder in enumerate(self.mask_builders):
            mask_builder.update(tiles[..., i], metadata["x"], metadata["y"])

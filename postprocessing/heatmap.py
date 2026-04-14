from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence

import mlflow
import numpy as np
import pandas as pd
import pyvips
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from lymph_nodes.data.data_module import collate_fn


HEATMAP_FACTOR: int = 16


def assemble_heatmap(
    tiles: pd.DataFrame,
    tile_extent: int = 224,
    factor: int = HEATMAP_FACTOR,
) -> pyvips.Image:
    norm_extent: int = max(1, tile_extent // factor)

    xs = tiles["tile_x"].to_numpy(dtype=np.int64) // factor
    ys = tiles["tile_y"].to_numpy(dtype=np.int64) // factor
    probs = tiles["prob"].to_numpy(dtype=np.float32)

    h = int(ys.max()) + norm_extent + 1
    w = int(xs.max()) + norm_extent + 1

    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.uint32)

    for x, y, prob in zip(xs, ys, probs, strict=True):
        heatmap[y : y + norm_extent, x : x + norm_extent] += prob
        counts[y : y + norm_extent, x : x + norm_extent] += 1

    safe_counts = np.where(counts == 0, 1, counts)
    averaged = (heatmap / safe_counts * 255).astype(np.uint8)

    return pyvips.Image.new_from_array(averaged)


class HeatmapWriter(BasePredictionWriter):
    def __init__(
        self,
        dest: str,
        tile_extent: int = 224,
        factor: int = HEATMAP_FACTOR,
    ) -> None:
        super().__init__(write_interval="epoch")
        self.dest = Path(dest)
        self.tile_extent = tile_extent
        self.factor = factor

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any] | None,
    ) -> None:
        if predictions and isinstance(predictions[0], list):
            all_batches: list[dict] = [
                batch for dl_preds in predictions for batch in dl_preds
            ]
        else:
            all_batches = list(predictions)

        rows: list[dict] = []
        for batch_preds in all_batches:
            probs = batch_preds["probs"].cpu().numpy()
            metadatas = batch_preds["metadatas"]
            for prob, meta in zip(probs, metadatas, strict=True):
                rows.append(
                    {
                        "slide_name": meta["slide_name"],
                        "tile_x": int(meta["tile_x"]),
                        "tile_y": int(meta["tile_y"]),
                        "prob": float(prob),
                    }
                )

        if not rows:
            return

        df = pd.DataFrame(rows)
        self.dest.mkdir(parents=True, exist_ok=True)

        for slide_name, slide_df in df.groupby("slide_name"):
            heatmap = assemble_heatmap(
                slide_df.reset_index(drop=True),
                tile_extent=self.tile_extent,
                factor=self.factor,
            )
            out_path = self.dest / f"{slide_name}.tiff"
            heatmap.tiffsave(
                str(out_path),
                compression="deflate",
                tile=True,
                tile_width=256,
                tile_height=256,
                bigtiff=True,
                pyramid=True,
            )

        mlflow.log_artifacts(str(self.dest), artifact_path="heatmaps")


class ValHeatmapCallback(Callback):
    """Runs predict on val slides after fit and saves heatmaps to val_heatmaps."""

    def __init__(
        self,
        dest: str,
        tile_extent: int = 224,
        factor: int = HEATMAP_FACTOR,
        batch_size: int = 512,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.dest = Path(dest)
        self.tile_extent = tile_extent
        self.factor = factor
        self.batch_size = batch_size
        self.num_workers = num_workers

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        val_dl = DataLoader(
            trainer.datamodule.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        pl_module.eval()
        all_batches = trainer.predict(
            pl_module, dataloaders=val_dl, return_predictions=True
        )
        pl_module.train()

        if not all_batches:
            return

        rows: list[dict] = []
        for batch_preds in all_batches:
            probs = batch_preds["probs"].cpu().numpy()
            metadatas = batch_preds["metadatas"]
            for prob, meta in zip(probs, metadatas, strict=True):
                rows.append(
                    {
                        "slide_name": meta["slide_name"],
                        "tile_x": int(meta["tile_x"]),
                        "tile_y": int(meta["tile_y"]),
                        "prob": float(prob),
                    }
                )

        if not rows:
            return

        import pandas as pd

        df = pd.DataFrame(rows)
        self.dest.mkdir(parents=True, exist_ok=True)

        for slide_name, slide_df in df.groupby("slide_name"):
            heatmap = assemble_heatmap(
                slide_df.reset_index(drop=True),
                tile_extent=self.tile_extent,
                factor=self.factor,
            )
            out_path = self.dest / f"{slide_name}.tiff"
            heatmap.tiffsave(
                str(out_path),
                compression="deflate",
                tile=True,
                tile_width=256,
                tile_height=256,
                bigtiff=True,
                pyramid=True,
            )

        mlflow.log_artifacts(str(self.dest), artifact_path="val_heatmaps")

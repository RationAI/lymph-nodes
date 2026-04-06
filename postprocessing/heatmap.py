"""Heatmap visualisation for patch-level MLP predictions.

After a ``trainer.predict()`` run the ``HeatmapWriter`` callback groups
per-tile probabilities by slide, assembles an averaged probability map and
writes it as a tiled BigTIFF (uint8, 0-255 scale).

Coordinate convention
---------------------
Tile coordinates ``tile_x`` / ``tile_y`` are pixel offsets in the original
slide image space, as stored in the embedding parquet files.  The heatmap is
downscaled by ``HEATMAP_FACTOR`` (default 16) to produce a manageable image
size.  With the default tiling parameters (tile_extent=224, stride=112) each
tile occupies a 14x14 region in the heatmap; overlapping tiles are averaged.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence

import mlflow
import numpy as np
import pandas as pd
import pyvips
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter


# Downscale factor applied to all coordinates when building the heatmap.
# With tile_extent=224 each tile maps to a 14x14-pixel region in the heatmap.
HEATMAP_FACTOR: int = 16


def assemble_heatmap(
    tiles: pd.DataFrame,
    tile_extent: int = 224,
    factor: int = HEATMAP_FACTOR,
) -> pyvips.Image:
    """Assemble a single-slide probability heatmap from per-tile predictions.

    Args:
        tiles:
            DataFrame with columns ``tile_x``, ``tile_y`` (pixel coordinates
            in the original image space) and ``prob`` (predicted probability
            in [0, 1]).
        tile_extent:
            Size of each tile in original image pixel space.
        factor:
            Downscaling factor applied to all coordinates and extents.

    Returns:
        A single-band ``pyvips.Image`` with dtype uint8, range [0, 255],
        where 255 corresponds to probability 1.0.
    """
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
    """Lightning callback that assembles per-slide heatmaps after prediction.

    Attach this callback to the Trainer when running in ``predict`` mode.
    After all batches are processed, per-tile probabilities are grouped by
    slide, assembled into an averaged probability image, written as a tiled
    BigTIFF to ``dest``, and logged to the active MLflow run under the
    ``heatmaps`` artifact path.

    Args:
        dest:
            Local directory where TIFF heatmaps will be written.
        tile_extent:
            Tile size in the original image coordinate space (pixels).
            Must match the value used during tiling.  Defaults to 224.
        factor:
            Downscaling factor applied to all coordinates when building
            the heatmap image.  Defaults to 16.
    """

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

    # Required by the ABC but not used — we collect everything at epoch end.
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
        # Lightning wraps predictions in one list per dataloader.
        # Handle both the flat list (single DL) and list-of-lists cases.
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

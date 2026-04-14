from pathlib import Path

import mlflow
from heatmap import HEATMAP_FACTOR, assemble_heatmap
from lightning import Callback, LightningModule, Trainer


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
        from torch.utils.data import DataLoader

        from lymph_nodes.data.data_module import collate_fn

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

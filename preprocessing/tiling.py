import tempfile
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.utils import row_hash

from preprocessing.parquet_dataset import from_parquet


# ── tile generation ───────────────────────────────────────────────────────────

def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "path": row["path"],
        "slide_id": row["id"],
        "mpp_x": row["mpp_x"],
        "mpp_y": row["mpp_y"],
        "tile_extent_x": row["tile_extent_x"],
        "tile_extent_y": row["tile_extent_y"],
    }
    return [
        {"tile_x": int(x), "tile_y": int(y), **common}
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


# ── entrypoint ────────────────────────────────────────────────────────────────

@with_cli_args(["+preprocessing=tiling"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_csv = pd.read_csv(download_artifacts(config.slides_uri))
    slide_paths = slides_csv["slide_path"].tolist()[:2]
    meta_cols = [c for c in slides_csv.columns if c != "slide_path"]
    slides_meta: dict[str, dict] = {
        row["slide_path"]: {k: row[k] for k in meta_cols}
        for _, row in slides_csv.iterrows()
    }

    # --- Slide-level Ray Dataset ---
    slides = read_slides(slide_paths, mpp=config.mpp, tile_extent=config.tile_extent, stride=config.stride)
    slides = slides.map(row_hash)
    slides = slides.map(
        lambda r, meta=slides_meta: {**r, **meta.get(r["path"], {})},
    )

    # --- Tile grid ---
    tiles = (
        slides
        .flat_map(tiling)
        .repartition(target_num_rows_per_block=512)
    )

    # --- Tiling blocks (sequential, config order) ---
    for block_conf in config.tiling_blocks:
        block = hydra.utils.instantiate(block_conf, _recursive_=True)
        tiles = block.apply(tiles)
    

    # --- Write sharded parquet + log to MLflow ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiles_dir = Path(tmp_dir) / "tiles"
        slides_dir = Path(tmp_dir) / "slides"
        tiles_dir.mkdir()
        slides_dir.mkdir()

        tiles.repartition(target_num_rows_per_block=config.rows_per_shard).write_parquet(str(tiles_dir))
        slides.write_parquet(str(slides_dir))

        logger.log_artifacts(tmp_dir)

        mlflow.log_input(
            from_parquet(str(tiles_dir), name=f"{config.dataset.name}_tiles"),
            context="tiles",
        )
        mlflow.log_input(
            from_parquet(str(slides_dir), name=f"{config.dataset.name}_slides"),
            context="slides",
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

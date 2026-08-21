import tempfile
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.utils import row_hash

from preprocessing.parquet_dataset import from_parquet
from preprocessing.tiling_blocks.tiling_block import TilingBlock


OmegaConf.register_new_resolver("scale", lambda x, factor: x * factor)


# ── tile generation ───────────────────────────────────────────────────────────

def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "path": row["path"],
        "slide_id": row["id"],
        "mpp_x": row["mpp_x"],
        "mpp_y": row["mpp_y"],
        "tile_extent_x": row["tile_extent_x"],
        "tile_extent_y": row["tile_extent_y"],
        "level": row["level"],
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


def tile_dataset(
    slides_df: pd.DataFrame,
    tiling_blocks: list[TilingBlock],
    tile_extent: tuple[int, int],
    stride: tuple[int, int],
    mpp: int,
    rows_per_shard: int,
    dataset_name: str,
    logger: MLFlowLogger,
) -> None:
    slide_paths = slides_df["slide_path"].tolist()
    meta_cols = [c for c in slides_df.columns if c != "slide_path"]

    slides_meta: dict[str, dict] = {
        row["slide_path"]: {k: row[k] for k in meta_cols}
        for _, row in slides_df.iterrows()
    }
    
    # --- Slide-level Ray Dataset ---
    slides = read_slides(slide_paths, mpp=mpp, tile_extent=tile_extent, stride=stride)
    slides = slides.map(row_hash)
    slides = slides.map(
        lambda r, meta=slides_meta: {**r, **meta.get(r["path"], {})},
    ).materialize()

    # --- Tile grid ---
    tiles = (
        slides
        .flat_map(tiling)
        .repartition(target_num_rows_per_block=512)
    )
    
    # --- Tiling blocks (sequential, config order) ---
    for block in tiling_blocks:
        tiles = block.apply(tiles)
        
    
    # --- Write sharded parquet + log to MLflow ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiles_dir = Path(tmp_dir) / dataset_name / "tiles"
        slides_dir = Path(tmp_dir) / dataset_name/ "slides"
        tiles_dir.mkdir(parents=True)
        slides_dir.mkdir(parents=True)

        tiles.repartition(target_num_rows_per_block=rows_per_shard).write_parquet(str(tiles_dir))
        slides.write_parquet(str(slides_dir))

        logger.log_artifacts(tmp_dir)

        mlflow.log_input(
            from_parquet(str(tiles_dir), name=f"{dataset_name}"),
            context="tiles",
        )
        mlflow.log_input(
            from_parquet(str(slides_dir), name=f"{dataset_name}"),
            context="slides",
        )


# ── entrypoint ────────────────────────────────────────────────────────────────

@with_cli_args(["+preprocessing=tiling"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    ray.init()
    tiling_blocks = [hydra.utils.instantiate(block_conf, _recursive_=False) for block_conf in config.tiling_blocks]

    for i, dataset in enumerate(config.dataset):
        print(f"Tiling dataset: {dataset.name} ({i + 1}/{len(config.dataset)})")
        slides_df = hydra.utils.instantiate(dataset.slides).to_pandas()
        tile_dataset(
            slides_df=slides_df,
            tiling_blocks=tiling_blocks,
            tile_extent=config.tile_extent,
            stride=config.stride,
            mpp=config.mpp,
            rows_per_shard=config.rows_per_shard,
            dataset_name=dataset.name,
            logger=logger,
        )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

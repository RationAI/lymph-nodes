import os
import tempfile
from typing import Any

import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import (
    grid_tiles,
    tile_overlay_overlap,
)
from ratiopath.tiling.utils import row_hash
from ray.data.expressions import col


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    tiles = []
    for x, y in grid_tiles(
        slide_extent=(row["extent_x"], row["extent_y"]),
        tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
        stride=(row["stride_x"], row["stride_y"]),
        last="keep",
    ):
        tiles.append(
            {
                "tile_x": x,
                "tile_y": y,
                "path": row["path"],
                "slide_id": row["id"],
                "mpp_x": row["mpp_x"],
                "mpp_y": row["mpp_y"],
                "tile_extent_x": row["tile_extent_x"],
                "tile_extent_y": row["tile_extent_y"],
                # mask paths propagated per-tile
                "tissue_mask_path": row["tissue_mask_path"],
                "blur_mask_path": row["blur_mask_path"],
            }
        )
    return tiles


def extract_coverage(row: dict[str, Any]) -> dict[str, Any]:
    # Histogram keyed by class label (255 = FG)
    row["tissue_coverage"] = row.get("tissue_overlap", {}).get(255, 0.0)
    row["blur_coverage"] = row.get("blur_overlap", {}).get(255, 0.0)
    return row


@hydra.main(
    config_path="../configs", config_name="preprocessing/tiling", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger):
    slides_list = hydra.utils.instantiate(config.dataset.slides)
    if not isinstance(slides_list, list):
        raise RuntimeError("dataset.slides must return a list of slide paths.")

    print(f"[INFO] Loaded {len(slides_list)} slide paths from ListDataSource")

    slides_ds = read_slides(
        path=slides_list,
        mpp=config.mpp,  # fixed: 0.5
        tile_extent=config.tile_extent,  # fixed: 224
        stride=config.stride,  # fixed: 112
    )

    # Unique slide ID
    slides_ds = slides_ds.map(
        row_hash,
        num_cpus=0.1,
        memory=128 * 1024**2,
    )

    tissue_dir = config.tissue_mask_dir
    blur_dir = config.blur_mask_dir

    def add_mask_paths(row):
        base = os.path.splitext(os.path.basename(row["path"]))[0]
        row["tissue_mask_path"] = os.path.join(tissue_dir, f"{base}_tissue_mask.tiff")
        row["blur_mask_path"] = os.path.join(blur_dir, f"{base}_blur_mask.tiff")
        return row

    slides_ds = slides_ds.map(add_mask_paths)

    tiles = slides_ds.flat_map(
        tiling,
        num_cpus=0.2,
        memory=128 * 1024**2,
    ).repartition(target_num_rows_per_block=128)

    full_roi = None

    tiles = tiles.add_column(
        "tissue_overlap",
        tile_overlay_overlap(
            full_roi,
            col("tissue_mask_path"),
            col("tile_x"),
            col("tile_y"),
            col("mpp_x"),
            col("mpp_y"),
        ),
    )

    tiles = tiles.add_column(
        "blur_overlap",
        tile_overlay_overlap(
            full_roi,
            col("blur_mask_path"),
            col("tile_x"),
            col("tile_y"),
            col("mpp_x"),
            col("mpp_y"),
        ),
    )

    tiles = tiles.map(extract_coverage)

    # Only remove tiles with ZERO tissue coverage
    tiles = tiles.filter(lambda r: r["tissue_coverage"] > 0)

    tiles = tiles.drop_columns(
        [
            "tissue_mask_path",
            "blur_mask_path",
            "tissue_overlap",
            "blur_overlap",
        ]
    )

    slides_df = slides_ds.to_pandas()
    tiles_df = tiles.to_pandas()

    with tempfile.TemporaryDirectory(dir=config.shared_dir, prefix="tiling_") as tmpdir:
        save_mlflow_dataset(
            slides=slides_df,
            tiles=tiles_df,
            dataset_name=config.dataset.name,
            output_dir=tmpdir,
        )

    print(f"[INFO] Completed. Slides: {len(slides_df)}, Tiles: {len(tiles_df)}")


if __name__ == "__main__":
    main()

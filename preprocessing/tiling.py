import os
import tempfile
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import (
    grid_tiles,
    overlay_roi,
    read_slide_tiles,
    tile_overlay_overlap,
)
from ratiopath.tiling.utils import row_hash
from ray.data.expressions import col
from shapely import Polygon

MIN_TISSUE_COVERAGE = 0.5
# Replace overlay_roi with a simple centered ROI using shapely Polygon (in tile space)
CENTERED_ROI = Polygon([
    (0.25, 0.25),
    (0.75, 0.25),
    (0.75, 0.75),
    (0.25, 0.75),
])

def download_masks(config: DictConfig) -> tuple[str, str]:
    tissue_mask_dir = mlflow.artifacts.download_artifacts(
        run_id=config.get("run_with_tissue_masks_id"),  # TODO
        dst_path="./tissue_masks",
    )
    blur_mask_dir = mlflow.artifacts.download_artifacts(
        run_id=config.get("run_with_blur_masks_id"),  # TODO
        dst_path="./blur_masks",
    )
    return tissue_mask_dir, blur_mask_dir

def build_slide_dataset(slides_ds, config: DictConfig):
    slides_path = list(hydra.utils.instantiate(config.dataset.slides))
    ds = read_slides(
        path=slides_path,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    )

    slides_metadata = ds.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)

    return slides_metadata

def build_tiles_dataset(slides_ds, config: DictConfig):
    tiles = slides_ds.flat_map(
        tiling,
        num_cpus=0.2,
        memory=128 * 1024**2,
    ).repartition(
        target_num_rows_per_block=128
    )
    return tiles

def enrich_tiles_with_masks(tiles, tissue_mask_dir: str, blur_mask_dir: str):
    # Attach mask paths based on slide path; adjust extensions to your mask naming
    def add_mask_paths(batch: dict[str, Any]) -> dict[str, Any]:
        batch["tissue_mask_path"] = batch["path"].str.replace(".mrxs", ".tiff").str.replace(
            "/slides/", f"/{os.path.basename(tissue_mask_dir)}/"
        )
        batch["blur_mask_path"] = batch["path"].str.replace(".mrxs", ".tiff").str.replace(
            "/slides/", f"/{os.path.basename(blur_mask_dir)}/"
        )
        return batch

    tiles = tiles.map_batches(add_mask_paths)

    # Compute overlap dictionaries using column expressions
    tiles = tiles.with_column(
        "tissue_mask_overlap",
        tile_overlay_overlap(
            ROI=CENTERED_ROI,
            overlay_path=col("tissue_mask_path"),
            tile_x=col("tile_x"),
            tile_y=col("tile_y"),
            mpp_x=col("mpp_x"),
            mpp_y=col("mpp_y"),
        ),
        num_cpus=1,
        memory=4 * 1024**3,
    ).with_column(
        "blur_mask_overlap",
        tile_overlay_overlap(
            ROI=CENTERED_ROI,
            overlay_path=col("blur_mask_path"),
            tile_x=col("tile_x"),
            tile_y=col("tile_y"),
            mpp_x=col("mpp_x"),
            mpp_y=col("mpp_y"),
        ),
        num_cpus=1,
        memory=4 * 1024**3,
    )

    # Extract foreground coverage
    tiles = tiles.map(extract_foreground_coverage)

    return tiles

def read_and_filter_tiles(tiles, config: DictConfig):
    # Read underlying slide tile pixels (optional; for pixel-based filtering)
    tiles = tiles.with_column(
        "tile",
        read_slide_tiles(
            col("path"),
            col("tile_x"),
            col("tile_y"),
            col("tile_extent_x"),
            col("tile_extent_y"),
            col("level"),
        ),
        num_cpus=config.get("num_cpus_tiles", 1),
        memory=config.get("memory_tiles", 4 * 1024**3),
    )

    # Filter by tissue coverage
    min_tissue_cov = config.get("min_tissue_coverage", MIN_TISSUE_COVERAGE)
    tiles = tiles.filter(lambda row: row.get("tissue_coverage", 0.0) > min_tissue_cov)

    # Drop heavy / intermediate columns if present
    cols_to_drop = []
    for col_name in ["tile", "tissue_mask_overlap", "blur_mask_overlap"]:
        if col_name in tiles.schema().names:
            cols_to_drop.append(col_name)
    if cols_to_drop:
        tiles = tiles.drop_columns(cols_to_drop)

    return tiles

def compute_overlaps(batch: dict[str, Any]) -> dict[str, Any]:
    """Compute overlap histograms for both masks"""
    batch["tissue_mask_overlap"] = tile_overlay_overlap(
        batch=batch,
        overlay_path_key="tissue_mask_path",
        roi=CENTERED_ROI,
    )
    batch["blur_mask_overlap"] = tile_overlay_overlap(
        batch=batch,
        overlay_path_key="blur_mask_path",
        roi=CENTERED_ROI,
    )
    return batch


def extract_foreground_coverage(row: dict[str, Any]) -> dict[str, Any]:
    """Extract tissue and blur coverage."""
    if "tissue_mask_overlap" in row:
        row["tissue_coverage"] = row["tissue_mask_overlap"].get(255, 0.0)
    if "blur_mask_overlap" in row:
        row["blur_coverage"] = row["blur_mask_overlap"].get(255, 0.0)
    return row


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Basic tiling: generate tile coordinates & metadata."""
    tiles: list[dict[str, Any]] = []
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
                "slide_id": row.get("slide_id", row.get("id")),
                "level": row["level"],
                "tile_extent_x": row["tile_extent_x"],
                "tile_extent_y": row["tile_extent_y"],
                "mpp_x": row["mpp_x"],
                "mpp_y": row["mpp_y"],
            }
        )
    return tiles


@hydra.main(
    config_path="../configs",
    config_name="preprocessing/tiling",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tissue_mask_dir, blur_mask_dir = download_masks(config)
    slides_ds = build_slide_dataset(None, config)
    tiles_ds = build_tiles_dataset(slides_ds, config)
    tiles_with_masks = enrich_tiles_with_masks(tiles_ds, tissue_mask_dir, blur_mask_dir)
    filtered_tiles = read_and_filter_tiles(tiles_with_masks, config)

    with tempfile.TemporaryDirectory(dir=config.shared_dir, prefix="tiling_") as tmpdir:
        slides_out = os.path.join(tmpdir, "slides")
        tiles_out = os.path.join(tmpdir, "tiles")
        slides_ds.write_parquet(slides_out)
        filtered_tiles.write_parquet(tiles_out)

        mlflow.set_experiment(experiment_name="Lymph Nodes")
        with mlflow.start_run(run_id="HDAB dataset tiling") as _:
            save_mlflow_dataset(
                slides=slides_ds,
                tiles=filtered_tiles,
                dataset_name="HDAB dataset - tiling",
            )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

"""
Example usage:
> uv run -m preprocessing.tiling +experiment=<experiment_name>
"""

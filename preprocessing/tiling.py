from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.parsers import ASAPParser
from ratiopath.ray import read_slides
from ratiopath.tiling import (
    grid_tiles,
    read_slide_tiles,
    tile_annotations,
    tile_overlay,
    tile_overlay_overlap,
)
from ratiopath.tiling.utils import row_hash
from shapely import Polygon


def add_overlay_tissue_mask_path(batch: dict) -> dict:
    """Add tissue mask overlay path for each tile in the batch."""
    batch["tissue_mask_path"] = batch["path"].str.replace(".mrxs", "_tissue_mask.tiff")
    return batch


def add_overlay_blur_mask_path(batch: dict) -> dict:
    """Add blur mask overlay path for each tile in the batch."""
    batch["blur_mask_path"] = batch["path"].str.replace(".mrxs", "_blur_mask.tiff")
    return batch


def extract_foreground_coverage(tile: dict) -> dict:
    """Extract tissue and blur coverage from overlay overlaps."""
    tile["tissue_coverage"] = tile["tissue_mask_overlap"].get(255, 0.0)
    tile["blur_coverage"] = tile["blur_mask_overlap"].get(255, 0.0)
    return tile


def tiling_with_annotations(row: dict[str, Any]) -> list[dict[str, Any]]:
    annotation_path = row["path"].replace(".mrxs", ".xml")
    parser = ASAPParser(annotation_path)
    annotations = list(parser.get_polygons(name="...", part_of_group="..."))

    roi = Polygon(
        [
            (0, 0),
            (row["tile_extent_x"], 0),
            (row["tile_extent_x"], row["tile_extent_y"]),
            (0, row["tile_extent_y"]),
        ]
    )

    coordinates = np.array(
        list(
            grid_tiles(
                slide_extent=(row["extent_x"], row["extent_y"]),
                tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
                stride=(row["stride_x"], row["stride_y"]),
                last="keep",
            )
        )
    )
    return [
        {
            "tile_x": coordinates[i, 0],
            "tile_y": coordinates[i, 1],
            "path": row["path"],
            "slide_id": row["id"],
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
            "coverage": polygon.area / roi.area,
        }
        for i, polygon in enumerate(
            tile_annotations(
                annotations,
                roi,
                coordinates,
                row["downsample"],
            )
        )
    ]


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["slide_id"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


@hydra.main(
    config_path="../configs", config_name="preprocessing/tiling", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # Read slides metadata
    slides = hydra.utils.instantiate(config.dataset.slides)
    slide_paths = [str(slide) for slide in slides]

    # Read slides with tiling parameters
    slides_ray = read_slides(
        path=slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    )

    # Add slide hash for tracking
    slides_ray = slides_ray.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)
    slides_ray.write_parquet(f"{config.output_dir}/slides")

    # Generate tiles
    if config.get("with_annotations", False):
        tiles = slides_ray.flat_map(
            tiling_with_annotations, num_cpus=0.2, memory=128 * 1024**2
        )
    else:
        tiles = slides_ray.flat_map(tiling, num_cpus=0.2, memory=128 * 1024**2)

    tiles = tiles.repartition(target_num_rows_per_block=128)

    # Add tissue mask overlays
    if config.get("with_tissue_mask", False):
        tiles = tiles.map_batches(add_overlay_tissue_mask_path)

        add_tissue_mask = tile_overlay(
            overlay_path_key="tissue_mask_path",
            store_key="tissue_mask",
        )
        tiles = tiles.map_batches(add_tissue_mask)

        add_tissue_mask_overlap = tile_overlay_overlap(
            overlay_key="tissue_mask",
            store_key="tissue_mask_overlap",
        )
        tiles = tiles.map_batches(add_tissue_mask_overlap)
        tiles = tiles.map(extract_foreground_coverage)

    # Add blur mask overlays
    if config.get("with_blur_mask", False):
        tiles = tiles.map_batches(add_overlay_blur_mask_path)

        add_blur_mask = tile_overlay(
            overlay_path_key="blur_mask_path",
            store_key="tissue_mask",
        )
        tiles = tiles.map_batches(add_blur_mask)

        add_blur_mask_overlap = tile_overlay_overlap(
            overlay_key="blur_mask",
            store_key="blue_mask_overlap",
        )
        tiles = tiles.map_batches(add_blur_mask_overlap)

    # Read tile images and filter low-variance tiles
    tiles = tiles.map_batches(read_slide_tiles, num_cpus=2, memory=2)

    # Filter out background tiles
    tiles = tiles.filter(lambda row: row["tile"].std() > config.get("std_threshold", 8))

    # Apply QC filters
    if config.get("with_tissue_mask", True):
        tiles = tiles.filter(
            lambda row: row["tissue_coverage"] > config.get("min_tissue_coverage", 0.5)
        )

    if config.get("use_blur_masks", True):
        tiles = tiles.filter(
            lambda row: row["blur_coverage"] < config.get("max_blur_coverage", 0.3)
        )

    # Drop unnecessary columns
    columns_to_drop = ["tile", "tissue_mask", "blur_mask"]
    tiles = tiles.drop_columns(
        [col for col in columns_to_drop if col in tiles.schema().names]
    )

    output_path = f"{config.output_dir}/tiles"
    tiles.write_parquet(output_path)
    logger.log_artifacts(config.output_dir, artifact_path=config.artifact_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


######################
##### RUN PARAMS #####
######################
"""
# Basic tiling without annotations
uv run -m preprocessing.tiling +experiment=data_sources/mmci_hdab_test_inference

# With annotations
uv run -m preprocessing.tiling +experiment=data_sources/mmci_hdab_test_inference use_annotations=true

# Adjust quality thresholds
uv run -m preprocessing.tiling +experiment=data_sources/mmci_hdab_test_inference min_tissue_coverage=0.7 max_blur_coverage=0.2

GPU: None
CPU: 2
RAM: 2Gi
"""

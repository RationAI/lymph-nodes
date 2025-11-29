from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.parsers import ASAPParser
from ratiopath.ray import read_slides
from ratiopath.tiling import (
    grid_tiles,
    overlay_roi,
    read_slide_tiles,
    tile_annotations,
    tile_overlay,
    tile_overlay_overlap,
)
from ratiopath.tiling.utils import row_hash
from shapely import Polygon


CENTERED_ROI = overlay_roi(
    offset_x_frac=0.25,  # 25 % from left
    offset_y_frac=0.25,  # 25 % from top
    extent_x_frac=0.5,  # 50 % of width
    extent_y_frac=0.5,  # 50 % of height
)


def add_overlay_tissue_mask_path(df: pd.DataFrame) -> pd.DataFrame:
    """Add tissue mask overlay path for each tile in the batch."""
    df = df.copy()
    df["tissue_mask_path"] = df["path"].str.replace(".mrxs", "_tissue_mask.tiff")
    return df


def add_overlay_blur_mask_path(df: pd.DataFrame) -> pd.DataFrame:
    """Add blur mask overlay path for each tile in the batch."""
    df = df.copy()
    df["blur_mask_path"] = df["path"].str.replace(".mrxs", "_blur_mask.tiff")
    return df


def add_tissue_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Read tissue mask overlay tiles and attach as 'tissue_mask' column."""
    df = df.copy()
    overlays = tile_overlay(
        batch=df.to_dict(orient="list"),
        overlay_path_key="tissue_mask_path",
        roi=CENTERED_ROI,
    )
    df["tissue_mask"] = overlays
    return df


def add_blur_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Read blur mask overlay tiles and attach as 'blur_mask' column."""
    df = df.copy()
    overlays = tile_overlay(
        batch=df.to_dict(orient="list"),
        overlay_path_key="blur_mask_path",
        roi=CENTERED_ROI,
    )
    df["blur_mask"] = overlays
    return df


def add_tissue_mask_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overlap ratios for tissue mask and store as 'tissue_mask_overlap'."""
    df = df.copy()
    overlaps = tile_overlay_overlap(
        batch=df.to_dict(orient="list"),
        overlay_path_key="tissue_mask_path",
        roi=CENTERED_ROI,
    )
    df["tissue_mask_overlap"] = overlaps
    return df


def add_blur_mask_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overlap ratios for blur mask."""
    df = df.copy()
    overlaps = tile_overlay_overlap(
        batch=df.to_dict(orient="list"),
        overlay_path_key="blur_mask_path",
        roi=CENTERED_ROI,
    )
    df["blur_mask_overlap"] = overlaps
    return df


def extract_foreground_coverage(row: dict[str, Any]) -> dict[str, Any]:
    """Extract tissue and blur coverage."""
    if "tissue_mask_overlap" in row:
        row["tissue_coverage"] = row["tissue_mask_overlap"].get(255, 0.0)
    if "blur_mask_overlap" in row:
        row["blur_coverage"] = row["blur_mask_overlap"].get(255, 0.0)
    return row


def tiling_with_annotations(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate tiles and compute annotation coverage per tile."""
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
            "annotation_coverage": polygon.area / roi.area if roi.area > 0 else 0.0,
        }
        for i, polygon in enumerate(
            tile_annotations(
                annotations=annotations,
                roi=roi,
                coordinates=coordinates,
                downsample=row["downsample"],
            )
        )
    ]


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
    # Read slides metadata
    slides_ray = read_slides(
        path=config.slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    )

    # Add unique hash ID for each slide
    slides_ray = slides_ray.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)
    slides_ray.write_parquet(f"{config.output_dir}/slides")

    # Tiling
    if config.get("with_annotations", False):
        tiles = slides_ray.flat_map(
            tiling_with_annotations,
            num_cpus=0.2,
            memory=128 * 1024**2,
        )
    else:
        tiles = slides_ray.flat_map(
            tiling,
            num_cpus=0.2,
            memory=128 * 1024**2,
        )

    tiles = tiles.repartition(target_num_rows_per_block=128)

    # Overlays: tissue mask
    if config.get("with_tissue_mask", False):
        tiles = tiles.map_batches(
            add_overlay_tissue_mask_path,
            batch_format="pandas",
        )

        tiles = tiles.map_batches(
            add_tissue_mask,
            batch_format="pandas",
        )

        tiles = tiles.map_batches(
            add_tissue_mask_overlap,
            batch_format="pandas",
        )

    # Overlays: blur mask
    if config.get("with_blur_mask", False):
        tiles = tiles.map_batches(
            add_overlay_blur_mask_path,
            batch_format="pandas",
        )

        tiles = tiles.map_batches(
            add_blur_mask,
            batch_format="pandas",
        )

        tiles = tiles.map_batches(
            add_blur_mask_overlap,
            batch_format="pandas",
        )

    # Extract coverage values from overlap dicts
    if config.get("with_tissue_mask", False) or config.get("with_blur_mask", False):
        tiles = tiles.map(extract_foreground_coverage)

    # Read tile images and filter low-variance tiles
    tiles = tiles.map_batches(
        read_slide_tiles,
        num_cpus=config.get("num_cpus_tiles", 1),
        memory=config.get("memory_tiles", 4 * 1024**3),
    )

    # Filter out low-variance (background) tiles
    std_threshold = config.get("std_threshold", 8.0)
    tiles = tiles.filter(lambda row: row["tile"].std() > std_threshold)

    # QC filters using masks
    if config.get("with_tissue_mask", False):
        min_tissue_cov = config.get("min_tissue_coverage", 0.5)
        tiles = tiles.filter(
            lambda row: row.get("tissue_coverage", 0.0) > min_tissue_cov
        )

    if config.get("with_blur_mask", False):
        max_blur_cov = config.get("max_blur_coverage", 0.3)
        tiles = tiles.filter(lambda row: row.get("blur_coverage", 0.0) < max_blur_cov)

    # Drop heavy / intermediate columns
    cols_to_drop = []
    for col in [
        "tile",
        "tissue_mask",
        "blur_mask",
        "tissue_mask_overlap",
        "blur_mask_overlap",
    ]:
        if col in tiles.schema().names:
            cols_to_drop.append(col)

    if cols_to_drop:
        tiles = tiles.drop_columns(cols_to_drop)

    # Write output and log to MLflow
    output_path = f"{config.output_dir}/tiles"
    tiles.write_parquet(output_path)

    logger.log_artifacts(
        local_dir=config.output_dir,
        artifact_path=config.artifact_path,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

"""
Example usage:
python preprocessing/tiling.py slide_paths=/path/to/slides/ output_dir=/path/to/output
"""

import os
import tempfile
from typing import Any

import hydra
import mlflow
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
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


CENTERED_ROI = overlay_roi(
    offset_x_frac=0.25,  # 25 % from left
    offset_y_frac=0.25,  # 25 % from top
    extent_x_frac=0.5,  # 50 % of width
    extent_y_frac=0.5,  # 50 % of height
)


# def add_overlay_mask_paths(batch: dict[str, Any], tissue_mask_dir: str, blur_mask_dir: str) -> dict[str, Any]:
#     """Add tissue and blur mask overlay paths for each tile in the batch."""
#     names = [os.path.splitext(os.path.basename(p))[0] for p in batch["path"]]
#     batch["tissue_mask_path"] = [f"{tissue_mask_dir}/{n}_tissue_mask.tiff" for n in names]
#     batch["blur_mask_path"] = [f"{blur_mask_dir}/{n}_blur_mask.tiff" for n in names]
#     return batch


# TODO : not used anywhere
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
    tissue_mask_path = mlflow.artifacts.download_artifacts(
        run_id=run_with_tissue_masks_id,  # TODO
        dst_path="./tissue_masks",
    )
    blur_mask_path = mlflow.artifacts.download_artifacts(
        run_id=run_with_blur_masks_id,  # TODO
        dst_path="./blur_masks",
    )

    slides = list(hydra.utils.instantiate(config.dataset.slides))

    # Read slides metadata
    slides_ray = read_slides(
        path=config.slide_paths,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    )

    # Add unique hash ID for each slide
    slides_ray = slides_ray.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)

    # Tiling
    tiles = slides_ray.flat_map(
        tiling,
        num_cpus=0.2,
        memory=128 * 1024**2,
    )

    tiles = tiles.repartition(target_num_rows_per_block=128)

    # # Build overlay paths (replace entire path with mask directories) in one pass, no pandas
    # tiles = tiles.map_batches(
    #     add_overlay_mask_paths,
    #     fn_kwargs={"tissue_mask_dir": tissue_mask_path, "blur_mask_dir": blur_mask_path},
    #     batch_format="numpy",
    # )

    # # Compute overlays and overlap dicts for both masks in one pass, no pandas
    # tiles = tiles.map_batches(
    #     compute_overlays,
    #     batch_format="numpy",
    # )

    # Extract coverage values from overlap dicts
    tiles = tiles.map(extract_foreground_coverage)

    # Read tile images and filter low-variance tiles
    tiles = tiles.map_batches(
        read_slide_tiles,
        num_cpus=config.get("num_cpus_tiles", 1),
        memory=config.get("memory_tiles", 4 * 1024**3),
    )

    # QC filters using masks
    min_tissue_cov = config.get("min_tissue_coverage", 0.5)
    tiles = tiles.filter(lambda row: row.get("tissue_coverage", 0.0) > min_tissue_cov)

    # Drop heavy / intermediate columns
    cols_to_drop = []
    for col in [
        "tile",
        "tissue_mask_overlap",
        "blur_mask_overlap",
    ]:
        if col in tiles.schema().names:
            cols_to_drop.append(col)

    if cols_to_drop:
        tiles = tiles.drop_columns(cols_to_drop)

    with tempfile.TemporaryDirectory(dir=config.shared_dir, prefix="tiling_") as tmpdir:
        slides_out = os.path.join(tmpdir, "slides")
        tiles_out = os.path.join(tmpdir, "tiles")
        slides_ray.write_parquet(slides_out)
        tiles.write_parquet(tiles_out)

    # Log dataset to MLflow
    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_id="HDAB dataset tiling") as _:
        save_mlflow_dataset(
            slides=slides,
            tiles=tiles,
            dataset_name="HDAB dataset - tiling",
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

"""
Example usage:
> uv run -m preprocessing.tiling +experiment=<experiment_name>
"""

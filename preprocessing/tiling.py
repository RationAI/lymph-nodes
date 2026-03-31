import os
from pathlib import Path
from typing import Any

import hydra
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import (
    grid_tiles,
    tile_overlay_overlap,
)
from ratiopath.tiling.utils import row_hash
from ray.data.expressions import col
from shapely.geometry import box


def make_tissue_roi(tile_extent: int):
    offset = tile_extent // 4
    size = tile_extent // 2
    return box(offset, offset, offset + size, offset + size)


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    slide_extent = (row["extent_x"], row["extent_y"])
    tile_extent = (row["tile_extent_x"], row["tile_extent_y"])
    stride = (row["stride_x"], row["stride_y"])

    common = {
        "path": row["path"],
        "slide_id": row["id"],
        "mpp_x": row["mpp_x"],
        "mpp_y": row["mpp_y"],
        "tile_extent_x": row["tile_extent_x"],
        "tile_extent_y": row["tile_extent_y"],
        "tissue_mask_path": row["tissue_mask_path"],
        "blur_mask_path": row.get("blur_mask_path"),
        "cytokeratin_mask_path": row.get("cytokeratin_mask_path"),
    }

    return [
        {
            "tile_x": x,
            "tile_y": y,
            **common,
        }
        for x, y in grid_tiles(
            slide_extent=slide_extent,
            tile_extent=tile_extent,
            stride=stride,
            last="keep",
        )
    ]


def extract_coverage(row: dict[str, Any]) -> dict[str, Any]:
    cyto_overlap = row.get("cytokeratin_overlap", {}).get("255", 0.0) or 0.0

    return {
        **row,
        "tissue_coverage": 1.0 - (row.get("tissue_overlap", {}).get("0", 0.0) or 0.0),
        "blur_coverage": 1.0 - (row.get("blur_overlap", {}).get("0", 0.0) or 0.0),
        "metastazis": cyto_overlap,
    }


def tissue_coverage(overlap: dict[str, float]) -> float:
    return 1.0 - (overlap.get("0", 0.0) or 0.0)


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger=MLFlowLogger):
    slide_source = hydra.utils.instantiate(config.dataset.slides)
    slides_list = list(slide_source)

    slides_ds = read_slides(
        slides_list,
        mpp=config.mpp,
        tile_extent=config.tile_extent,
        stride=config.stride,
    )

    slides_ds = slides_ds.map(
        row_hash,
        num_cpus=0.1,
        memory=128 * 1024**2,
    )

    tissue_dir = download_artifacts(config.tissue_mask_uri)
    blur_dir = (
        download_artifacts(config.blur_mask_uri)
        if config.get("blur_mask_uri")
        else None
    )
    cytokeratin_dir = (
        download_artifacts(config.cytokeratin_mask_uri)
        if config.get("cytokeratin_mask_uri")
        else None
    )

    def add_mask_paths(row):
        filename = os.path.basename(row["path"])
        stem, _ = os.path.splitext(filename)

        tissue_path = os.path.join(tissue_dir, f"{stem}.tiff")
        if not os.path.exists(tissue_path):
            raise FileNotFoundError(f"Cannot find {tissue_path}")
        row["tissue_mask_path"] = tissue_path

        if cytokeratin_dir:
            possible_names = [f"{stem}.tiff", f"DAB-CK-{stem}.tiff"]
            found_cyto_path = None

            for name in possible_names:
                search_result = list(Path(cytokeratin_dir).rglob(name))
                if search_result:
                    found_cyto_path = str(search_result[0])
                    break

        if found_cyto_path:
            row["cytokeratin_mask_path"] = found_cyto_path
        else:
            print(f"No Cytokeratin mask found for {stem}. Assuming negative.")
            row["cytokeratin_mask_path"] = None

        if blur_dir:
            row["blur_mask_path"] = os.path.join(blur_dir, f"{stem}.tiff")

        return row

    slides_ds = slides_ds.map(add_mask_paths)

    tiles = slides_ds.flat_map(
        tiling,
        num_cpus=0.2,
        memory=512 * 1024**2,
    ).repartition(target_num_rows_per_block=512)

    tissue_roi = make_tissue_roi(config.tile_extent)

    tiles = tiles.with_column(
        "tissue_overlap",
        tile_overlay_overlap(
            tissue_roi,
            col("tissue_mask_path"),
            col("tile_x"),
            col("tile_y"),
            col("mpp_x"),
            col("mpp_y"),
        ),
        num_cpus=2,
        memory=2 * 3 * 512 * config.tile_extent**2,
    )

    tiles = tiles.filter(
        lambda r: tissue_coverage(r.get("tissue_overlap")) > config.min_tissue_coverage
    )

    if config.get("blur_mask_uri"):
        tiles = tiles.with_column(
            "blur_overlap",
            tile_overlay_overlap(
                tissue_roi,
                col("blur_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),
        )

    if config.get("cytokeratin_mask_uri"):
        tiles = tiles.with_column(
            "cytokeratin_overlap",
            tile_overlay_overlap(
                tissue_roi,
                col("cytokeratin_mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),
        )

    tiles = tiles.map(extract_coverage)

    columns_to_drop = [
        "tissue_mask_path",
        "tissue_overlap",
    ]
    if config.get("blur_mask_uri"):
        columns_to_drop.extend(["blur_mask_path", "blur_overlap"])
    if config.get("cytokeratin_mask_uri"):
        columns_to_drop.extend(["cytokeratin_mask_path", "cytokeratin_overlap"])

    tiles = tiles.drop_columns(columns_to_drop)

    slides_df = slides_ds.to_pandas()
    tiles_df = tiles.to_pandas()

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.dataset.name,
    )


if __name__ == "__main__":
    main()

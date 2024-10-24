from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import pandas as pd
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset

from preprocessing.data import (
    get_relative_dir_path,
    negative_training_wsis,
    positive_training_wsis,
)


TISSUE_MASKS_PATH = Path("data/tissue_masks")
ANNOTATION_MASKS_PATH = Path("data/annotation_masks")
RUN_WITH_MASKS = "e207076963d54cf58fd54df2b825019c"


@dataclass
class MetastazisTileMetadata(TileMetadata):
    metastazis_percentage: float


class TissueMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata | None:
        if class_overlaps.get(0, 0) > 0.95:
            return None
        return tile_labels


class MetastazisMask(PyvipsMask[MetastazisTileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> MetastazisTileMetadata:
        return MetastazisTileMetadata(
            **asdict(tile_labels), metastazis_percentage=class_overlaps.get(255, 0)
        )


source = OpenSlideTileSource(mpp=0.48, tile_extent=512, stride=256)
tissue_mask = TissueMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)
metastazis_mask = MetastazisMask(
    tile_extent=source.tile_extent, absolute_roi_extent=256, relative_roi_offset=0
)


@ray.remote
def positive_slide_handler(slide_path: Path) -> TiledSlideMetadata:
    tissue_mask_path = Path(
        TISSUE_MASKS_PATH, get_relative_dir_path(slide_path), f"{slide_path.stem}.tiff"
    )

    if not tissue_mask_path.exists():
        with open("tiler_missing_tissue_masks.txt", "a") as f:
            f.write(f"{slide_path}\n")
        return None

    cancer_mask_path = Path(
        ANNOTATION_MASKS_PATH,
        get_relative_dir_path(slide_path),
        f"{slide_path.stem}.tiff",
    )

    if not cancer_mask_path.exists():
        with open("tiler_missing_cancer_masks.txt", "a") as f:
            f.write(f"{slide_path}\n")
        return None

    slide, tiles = source(slide_path)
    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)
    tiles = metastazis_mask(cancer_mask_path, slide.extent, tiles)

    return slide, tiles


@ray.remote
def negative_slide_handler(slide_path: Path) -> TiledSlideMetadata:
    tissue_mask_path = Path(
        TISSUE_MASKS_PATH, get_relative_dir_path(slide_path), f"{slide_path.stem}.tiff"
    )

    if not tissue_mask_path.exists():
        with open("tiler_missing_tissue_masks.txt", "a") as f:
            f.write(f"{slide_path}\n")
        return None

    slide, tiles = source(slide_path)
    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)

    return slide, tiles


def tiler() -> None:
    # Downlaod artifacts
    mlflow.artifacts.download_artifacts(run_id=RUN_WITH_MASKS, dst_path="./data")

    positive_slides = positive_training_wsis()
    negative_slides = negative_training_wsis()

    negative_slides_df, negative_tiles_df = tiling(
        slides=list(negative_slides), handler=negative_slide_handler
    )
    positive_slides_df, positive_tiles_df = tiling(
        slides=list(positive_slides), handler=positive_slide_handler
    )

    negative_tiles_df["metastazis_percentage"] = 0.0
    negative_slides_df["kind"] = "lymp_nodes"
    positive_slides_df["kind"] = "colorectal"

    slides_df = pd.concat([negative_slides_df, positive_slides_df], ignore_index=True)
    tiles_df = pd.concat([negative_tiles_df, positive_tiles_df], ignore_index=True)

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB training data with epytelium") as _:
        save_mlflow_dataset(
            slides=slides_df,
            tiles=tiles_df,
            dataset_name="DAB Lymph Nodes with Epytelium - train",
        )

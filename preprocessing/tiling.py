from collections.abc import Iterable
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
    inference_wsis,
    negative_test_wsis,
    negative_train_wsis,
    negative_val_wsis,
    positive_train_wsis,
    positive_val_wsis,
    test_wsis_colorectal,
    test_wsis_lymph_nodes,
)


TISSUE_MASKS_PATH = Path("data/tissue_masks")
ANNOTATION_MASKS_PATH = Path("data/annotation_masks")


@dataclass
class MetastazisTileMetadata(TileMetadata):
    metastazis: float


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
            **asdict(tile_labels), metastazis=class_overlaps.get(255, 0)
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


def tile_dataset(
    positive_slides: Iterable[Path], negative_slides: Iterable[Path], kind: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_slides_df, positive_tiles_df = tiling(
        slides=list(positive_slides), handler=positive_slide_handler
    )

    negative_slides_df, negative_tiles_df = tiling(
        slides=list(negative_slides), handler=negative_slide_handler
    )

    negative_tiles_df["metastazis"] = 0.0
    negative_slides_df["kind"] = "lymph_nodes"
    positive_slides_df["kind"] = kind

    negative_slides_df["metastazis"] = False
    positive_slides_df["metastazis"] = True

    slides_df = pd.concat([negative_slides_df, positive_slides_df], ignore_index=True)
    tiles_df = pd.concat([negative_tiles_df, positive_tiles_df], ignore_index=True)

    return slides_df, tiles_df


def training_tiler(run_with_masks_id: str) -> None:
    # Downlaod artifacts
    mlflow.artifacts.download_artifacts(run_id=run_with_masks_id, dst_path="./data")

    train_slides, train_tiles = tile_dataset(
        positive_train_wsis(), negative_train_wsis(), "colorectal"
    )

    val_slides, val_tiles = tile_dataset(
        positive_val_wsis(), negative_val_wsis(), "colorectal"
    )

    # Testing data
    test_slides, test_tiles = tile_dataset(
        test_wsis_lymph_nodes(), negative_test_wsis(), "lymph_nodes"
    )

    test_slides_colorectal, test_tiles_colrectal = tile_dataset(
        test_wsis_colorectal(), [], "colorectal"
    )

    test_slides = pd.concat([test_slides, test_slides_colorectal], ignore_index=True)
    test_tiles = pd.concat([test_tiles, test_tiles_colrectal], ignore_index=True)

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB datasets with epitelium") as _:
        save_mlflow_dataset(
            slides=train_slides,
            tiles=train_tiles,
            dataset_name="DAB Lymph Nodes with Epitelium - train",
        )
        save_mlflow_dataset(
            slides=val_slides,
            tiles=val_tiles,
            dataset_name="DAB Lymph Nodes with Epitelium - val",
        )
        save_mlflow_dataset(
            slides=test_slides,
            tiles=test_tiles,
            dataset_name="DAB Lymph Nodes with Epitelium - test",
        )


def inference_tiler(run_with_masks_id: str) -> None:
    # Downlaod artifacts
    mlflow.artifacts.download_artifacts(run_id=run_with_masks_id, dst_path="./data")

    slides, tiles = tiling(
        slides=list(inference_wsis()), handler=negative_slide_handler
    )

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB inference dataset 2023") as _:
        save_mlflow_dataset(
            slides=slides,
            tiles=tiles,
            dataset_name="DAB dataset 2023 - inference",
        )

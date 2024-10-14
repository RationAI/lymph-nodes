from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import ray
from rationai.tiling import tiling
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset

from preprocessing.data import negative_training_wsis, positive_training_wsis


TISSUE_MASKS_PATH = Path("data/tissue_masks")
ANNOTATION_MASKS_PATH = Path("data/annotation_masks")


@dataclass
class MetastazisTileMetadata(TileMetadata):
    metastazis_percentage: float


class TissueMask(PyvipsMask[TileMetadata]):
    def forward_tile(
        self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
    ) -> TileMetadata | None:
        if class_overlaps.get(0, 0) > 0.5:
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
    slide, tiles = source(slide_path)

    tissue_mask_path = Path(TISSUE_MASKS_PATH, slide_path.name)
    cancer_mask_path = Path(ANNOTATION_MASKS_PATH, slide_path.name)

    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)
    tiles = metastazis_mask(cancer_mask_path, slide.extent, tiles)

    return slide, tiles


@ray.remote
def negative_slide_handler(slide_path: Path) -> TiledSlideMetadata:
    slide, tiles = source(slide_path)

    tissue_mask_path = Path(TISSUE_MASKS_PATH, slide_path.name)

    tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)

    return slide, tiles


def main() -> None:
    positive_slides = positive_training_wsis()
    negative_slides = negative_training_wsis()

    negative_slides_df, negative_tiles_df = tiling(
        slides=negative_slides, handler=negative_slide_handler
    )
    positive_slides_df, positive_tiles_df = tiling(
        slides=positive_slides, handler=positive_slide_handler
    )

    slides_df = negative_slides_df + positive_slides_df
    tiles_df = negative_tiles_df + positive_tiles_df

    mlflow.set_experiment(experiment_name="DAB Metastazis Detection")
    with mlflow.start_run(run_name="DAB training data with epytelium") as _:
        save_mlflow_dataset(
            slides=slides_df,
            tiles=tiles_df,
            dataset_name="DAB Lymph Nodes with Epytelium - train",
        )


if __name__ == "__main__":
    main()

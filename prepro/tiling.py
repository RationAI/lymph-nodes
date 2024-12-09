from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import pandas as pd
import ray
from rationai.tiling import tiling
from rationai.tiling.modules import TilingModule
from rationai.tiling.modules.masks import PyvipsMask
from rationai.tiling.modules.tile_sources import OpenSlideTileSource
from rationai.tiling.typing import TiledSlideMetadata, TileMetadata
from rationai.tiling.writers import save_mlflow_dataset

from prepro.utils import get_relative_dir_path


class Args(TypedDict):
    slides: Iterable[Path]
    desired_mpp: float
    tissue_threshold: float
    tile_extent: int
    stride: int
    source_kind: str
    slide_metastazis: bool
    tissue_masks_dir: str
    cytokeratin_masks_dir: str
    ignore_mask_dir: str
    annotation_masks_dir: str
    relative_path_prefix: str


@dataclass
class MetastazisTileMetadata(TileMetadata):
    metastazis: float


def data_tiler(
    slides: Iterable[Path],
    desired_mpp: float,
    tissue_threshold: float,
    tile_extent: int,
    stride: int,
    source_kind: str,
    slide_metastazis: bool,
    tissue_masks_dir: str,
    cytokeratin_masks_dir: str,
    ignore_mask_dir: str,
    annotation_masks_dir: str,
    relative_path_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Dedine tiling Modules
    class TissueMask(PyvipsMask[TileMetadata]):
        def forward_tile(
            self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
        ) -> TileMetadata | None:
            if class_overlaps.get(0, 0) > tissue_threshold:
                return None

            return tile_labels

    class MetastazisMask(PyvipsMask[MetastazisTileMetadata]):
        def forward_tile(
            self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
        ) -> MetastazisTileMetadata:
            return MetastazisTileMetadata(
                **asdict(tile_labels), metastazis=class_overlaps.get(255, 0)
            )

    class IgnoreMask(PyvipsMask[TileMetadata]):
        def forward_tile(
            self, tile_labels: TileMetadata, class_overlaps: dict[int, float]
        ) -> TileMetadata | None:
            if class_overlaps.get(255, 0) > 0:
                return None

            return tile_labels

    class NoMetastazisTiles(TilingModule):
        def forward(
            self, tiles_labels: Iterable[TileMetadata]
        ) -> list[MetastazisTileMetadata]:
            result: list[MetastazisTileMetadata] = []

            for tile_labels in tiles_labels:
                result.append(
                    MetastazisTileMetadata(**asdict(tile_labels), metastazis=0)
                )

            return result

    source = OpenSlideTileSource(
        mpp=desired_mpp, tile_extent=tile_extent, stride=stride
    )

    tissue_mask = TissueMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=tile_extent // 2,
        relative_roi_offset=0,
    )

    annotation_mask = MetastazisMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=tile_extent // 2,
        relative_roi_offset=0,
    )

    ignore_mask = IgnoreMask(
        tile_extent=source.tile_extent,
        absolute_roi_extent=tile_extent // 2,
        relative_roi_offset=0,
    )

    no_annotation_mask = NoMetastazisTiles()

    # Tiling function
    @ray.remote
    def slide_handler(slide_path: Path) -> TiledSlideMetadata:
        tissue_mask_path = Path(
            tissue_masks_dir,
            get_relative_dir_path(slide_path, Path(relative_path_prefix)),
            f"{slide_path.stem}.tiff",
        )

        annotation_mask_path = Path(
            annotation_masks_dir,
            get_relative_dir_path(slide_path, Path(relative_path_prefix)),
            f"{slide_path.stem}.tiff",
        )

        ignore_mask_path = Path(
            ignore_mask_dir,
            get_relative_dir_path(slide_path, Path(relative_path_prefix)),
            f"{slide_path.stem}.tiff",
        )

        cytokeratin_mask_path = Path(
            cytokeratin_masks_dir,
            get_relative_dir_path(slide_path, Path(relative_path_prefix)),
            f"{slide_path.stem}.tiff",
        )

        slide, tiles = source(slide_path)
        tiles = tissue_mask(tissue_mask_path, slide.extent, tiles)

        if ignore_mask_path.exists():
            tiles = ignore_mask(ignore_mask_path, slide.extent, tiles)

        if cytokeratin_mask_path.exists():
            annotated_tiles = annotation_mask(
                cytokeratin_mask_path, slide.extent, tiles
            )
        elif annotation_mask_path.exists():
            annotated_tiles = annotation_mask(annotation_mask_path, slide.extent, tiles)
        else:
            annotated_tiles = no_annotation_mask(tiles)

        return slide, annotated_tiles

    slides_df, tiles_df = tiling(slides=list(slides), handler=slide_handler)

    slides_df["kind"] = source_kind
    slides_df["metastazis"] = slide_metastazis

    return slides_df, tiles_df


def tile_dataset(sources: list[Args], dataset_name: str) -> None:
    slides_df = pd.DataFrame()
    tiles_df = pd.DataFrame()

    for args in sources:
        slides, tiles = data_tiler(**args)
        slides_df = pd.concat([slides_df, slides], ignore_index=True)
        tiles_df = pd.concat([tiles_df, tiles], ignore_index=True)

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=dataset_name,
    )

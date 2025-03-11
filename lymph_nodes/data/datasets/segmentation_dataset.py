import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import mlflow
import numpy as np
from numpy.typing import NDArray
from openslide import OpenSlide
from rationai.masks import (
    closest_level,
    slide_resolution,
)

from lymph_nodes.data.datasets.base_dataset import BaseDataset, SlideTiles
from lymph_nodes.typing import PredictSample, Sample


T = TypeVar("T", bound=Sample | PredictSample)


class MaskSlideTiles(SlideTiles[T]):
    def _get_mask(self, idx: int, mask_path: Path | str) -> NDArray | None:
        tile = self.slide_tiles.tiles.iloc[idx]

        if not os.path.exists(mask_path):
            return None

        with OpenSlide(mask_path) as slide:
            level = closest_level(slide, mpp=self.slide_metadata["mpp_x"])
            mpp_x, mpp_y = slide_resolution(slide, level)

            res_factor_x = self.slide_metadata["mpp_x"] / mpp_x
            res_factor_y = self.slide_metadata["mpp_y"] / mpp_y

            x = tile["x"] * res_factor_x
            y = tile["y"] * res_factor_y

            extent_x = tile["width"] * res_factor_x
            extent_y = tile["height"] * res_factor_y

            mask_x = int(x * slide.level_downsamples[level])
            mask_y = int(y * slide.level_downsamples[level])

            mask_extent_x = int(extent_x * slide.level_downsamples[level])
            mask_extent_y = int(extent_y * slide.level_downsamples[level])

            mask = slide.read_region(
                (mask_x, mask_y), level, (mask_extent_x, mask_extent_y)
            )
            return (
                np.array(mask.convert("L").resize(tile["height"], tile["width"])) / 255
            )

    def _get_mask_path(self, pref: str) -> Path:
        return Path(
            f"data/{pref}",
            Path(self.slide_metadata.path).relative_to(
                "/mnt/data/Projects/lymph_nodes"
            ),
        )


class _SegmentationSlideTiles(MaskSlideTiles[Sample]):
    def __getitem__(self, idx: int) -> Sample:
        np_image = self.slide_tiles[idx]

        cyto_path = self._get_mask_path("")
        annot_path = self._get_mask_path("annotation_masks")
        color_path = self._get_mask_path("color_separation_masks")

        cyto_mask = self._get_mask(idx, cyto_path)
        annot_mask = self._get_mask(idx, annot_path)
        color_mask = self._get_mask(idx, color_path)

        np_mask = cyto_mask if cyto_mask else np.min(annot_mask - color_mask, 0)

        metadata = self._get_metadata(idx)

        if self.transforms is not None:
            transformed = self.transforms(image=np_image, mask=np_mask)
            np_image = transformed["image"]
            np_mask = transformed["mask"]

        tensor = self.to_tensor(image=np_image, mask=np_mask)

        image = tensor["image"]
        mask = tensor["mask"]

        return image, mask, metadata


class _SegmentationSlideTilesPred(MaskSlideTiles[PredictSample]):
    def __getitem__(self, idx: int) -> PredictSample:
        np_image = self.slide_tiles[idx]
        metadata = self._get_metadata(idx)

        if self.transforms is not None:
            np_image = self.transforms(image=np_image)["image"]

        image = self.to_tensor(image=np_image)["image"]

        return image, metadata


class SegmentationDataset(BaseDataset[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: Any | None = None,
    ) -> None:
        super().__init__(
            uris=uris, sample_constructor=_SegmentationSlideTiles, transforms=transforms
        )

        # Color separation masks
        mlflow.artifacts.download_artifacts(
            artifact_uri="mlflow-artifacts:/68/c114214bc8c84b3191680cdbe6bc67d5/artifacts/color_separation_masks",
            dst_path="./data",
        )

        # Annotation masks
        mlflow.artifacts.download_artifacts(
            artifact_uri="mlflow-artifacts:/68/c114214bc8c84b3191680cdbe6bc67d5/artifacts/annotation_masks",
            dst_path="./data",
        )

        # Cytokeration masks
        mlflow.artifacts.download_artifacts(
            artifact_uri="mlflow-artifacts:/68/42ff2d1b9b8649bd94a4fe6360db201e/artifacts/cytokeratin_masks",
            dst_path="./data",
        )


class SegmentationPredictDataset(BaseDataset[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: Any | None = None,
    ) -> None:
        super().__init__(
            uris=uris,
            sample_constructor=_SegmentationSlideTilesPred,
            transforms=transforms,
        )


# class LymphNodesPredict(MetaTiledSlides[PredictSample]):
#     def __init__(
#         self,
#         uris: Iterable[str],
#         transforms: Any | None = None,
#     ) -> None:
#         self.transforms = transforms
#         super().__init__(uris=uris)

#     def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
#         slides_2023 = self.slides[self.slides["path"].str.contains("2023")]["id"]

#         self.tiles = self.tiles[
#             np.logical_and(
#                 self.tiles["slide_id"].isin(slides_2023),
#                 self.tiles["brownish"] > 0,
#             )
#         ].reset_index()

#         return (
#             _LymphNodesSlideTiles(
#                 slide,
#                 tiles=self.filter_tiles_by_slide(slide["id"]),
#                 include_label=False,
#                 transforms=self.transforms,
#             )
#             for _, slide in self.slides.iterrows()
#         )

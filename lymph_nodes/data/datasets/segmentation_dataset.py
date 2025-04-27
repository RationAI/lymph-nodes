import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from numpy.typing import NDArray
from openslide import OpenSlide
from rationai.masks import (
    closest_level,
    slide_resolution,
)

from lymph_nodes.data.datasets.base_dataset import BaseDataset, SlideTiles
from lymph_nodes.typing import PredictSample, SegSample


class MaskSlideTiles(SlideTiles[SegSample]):
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

            extent_x = self.slide_metadata["tile_extent_x"] * res_factor_x
            extent_y = self.slide_metadata["tile_extent_y"] * res_factor_y

            mask_x = round(x * slide.level_downsamples[level])
            mask_y = round(y * slide.level_downsamples[level])

            mask_extent_x = round(extent_x)
            mask_extent_y = round(extent_y)

            mask = slide.read_region(
                (mask_x, mask_y), level, (mask_extent_x, mask_extent_y)
            ).convert("1")

            return np.array(
                mask.resize(
                    (
                        self.slide_metadata["tile_extent_y"],
                        self.slide_metadata["tile_extent_x"],
                    )
                ),
                dtype=np.float32,
            )

    def _get_mask_path(self, pref: str) -> Path:
        return Path(
            f"data/{pref}",
            Path(self.slide_metadata.path).relative_to(
                "/mnt/data/Projects/lymph_nodes"
            ),
        )


class _SegmentationSlideTiles(MaskSlideTiles):
    def __getitem__(self, idx: int) -> SegSample:
        np_image = self.slide_tiles[idx]

        cyto_path = self._get_mask_path("cytokeratin_masks").with_suffix(".tiff")
        annot_path = self._get_mask_path("annotation_masks").with_suffix(".tiff")
        color_path = self._get_mask_path("color_separation_masks").with_suffix(".tiff")

        cyto_mask = self._get_mask(idx, cyto_path)
        annot_mask = self._get_mask(idx, annot_path)
        color_mask = self._get_mask(idx, color_path)

        np_mask = (
            cyto_mask
            if cyto_mask is not None
            else ((annot_mask if annot_mask is not None else 0) * color_mask)
        )

        metadata = self._get_metadata(idx)

        if self.transforms is not None:
            transformed = self.transforms(image=np_image, mask=np_mask)
            np_image = transformed["image"]
            np_mask = transformed["mask"]

        tensor = self.to_tensor(image=np_image, mask=np_mask)

        image = tensor["image"]
        mask = tensor["mask"]
        # label = torch.tensor([self.slide_tiles.tiles.iloc[idx]["cancer"]]).float()
        label = (mask.sum() > 0).float()

        return image, mask, label, metadata


class _SegmentationSlideTilesPred(SlideTiles[PredictSample]):
    def __getitem__(self, idx: int) -> PredictSample:
        np_image = self.slide_tiles[idx]
        metadata = self._get_metadata(idx)

        if self.transforms is not None:
            np_image = self.transforms(image=np_image)["image"]

        image = self.to_tensor(image=np_image)["image"]

        return image, metadata


class SegmentationDataset(BaseDataset[SegSample]):
    def __init__(
        self,
        uris: Iterable[str],
        slide_ids: list[str] | None = None,
        transforms: Any | None = None,
        no_neg_tma_tiles: bool = False,
        ignore_annotation: bool = True,
    ) -> None:
        super().__init__(
            uris=uris,
            sample_constructor=_SegmentationSlideTiles,
            transforms=transforms,
            slide_ids=slide_ids,
            no_neg_tma_tiles=no_neg_tma_tiles,
            ignore_annotation=ignore_annotation,
        )

        # Color separation masks
        if not os.path.exists("data/color_separation_masks"):
            mlflow.artifacts.download_artifacts(
                artifact_uri="mlflow-artifacts:/68/c862f7e8ff614c129a38aa83ae796de2/artifacts/color_separation_masks",
                dst_path="./data",
            )

        # Annotation masks
        if not os.path.exists("data/annotation_masks"):
            mlflow.artifacts.download_artifacts(
                artifact_uri="mlflow-artifacts:/68/c862f7e8ff614c129a38aa83ae796de2/artifacts/annotation_masks",
                dst_path="./data",
            )

        # Cytokeration masks
        if not os.path.exists("data/cytokeratin_masks"):
            mlflow.artifacts.download_artifacts(
                artifact_uri="mlflow-artifacts:/68/8ad173ea482d4db999bee7686d9dc2d5/artifacts/cytokeratin_masks",
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

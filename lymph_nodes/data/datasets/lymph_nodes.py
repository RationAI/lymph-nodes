from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from lymph_nodes.typing import Metadata, PredictSample, Sample


class LymphNodes(MetaTiledSlides[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        metastazis_threshold: float,
        transforms: Any | None = None,
    ) -> None:
        self.transforms = transforms
        self.metastazis_threshold = metastazis_threshold
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[Sample]]:
        # positive_slides_id = self.slides[self.slides["kind"] == "colorectal"]["id"]

        self.tiles["cancer"] = self.tiles["metastazis"] > self.metastazis_threshold
        # filter out all tiles that are from positive slides, but dont contain meteastazis (eg. < 0.5) then rest the index
        # self.tiles = self.tiles[
        #     ~np.logical_and(
        #         self.tiles["slide_id"].isin(positive_slides_id),
        #         ~self.tiles["cancer"],
        #     )
        # ].reset_index()

        return (
            _LymphNodesSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_label=True,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class LymphNodesPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: Any | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=uris)

    def generate_datasets(self) -> Iterable[Dataset[PredictSample]]:
        return (
            _LymphNodesSlideTiles(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_label=False,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )


class _LymphNodesSlideTiles(Dataset[Sample | PredictSample]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
        include_label: bool,
        transforms: Any | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata.path,
            level=slide_metadata.level,
            tile_extent_x=slide_metadata.tile_extent_x,
            tile_extent_y=slide_metadata.tile_extent_y,
            tiles=tiles,
        )
        self.transforms = transforms
        self.include_label = include_label
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> Sample | PredictSample:
        image = self.slide_tiles[idx]
        metadata = Metadata(
            slide_id=self.slide_tiles.tiles.iloc[idx]["slide_id"],
            slide=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]

        if self.include_label:
            label = torch.tensor([self.slide_tiles.tiles.iloc[idx]["cancer"]]).float()
            return image, label, metadata

        return image, metadata

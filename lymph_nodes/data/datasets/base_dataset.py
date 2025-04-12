from abc import ABC
from collections.abc import Iterable
from typing import Any, TypeVar

import pandas as pd
from albumentations.pytorch import ToTensorV2
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from lymph_nodes.typing import ClsSample, Metadata, PredictSample, SegSample


T = TypeVar("T", bound=SegSample | ClsSample | PredictSample)


class SlideTiles(Dataset[T]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: pd.DataFrame,
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
        self.slide_metadata = slide_metadata
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def _get_metadata(self, idx: int) -> Metadata:
        return Metadata(
            slide_id=self.slide_tiles.tiles.iloc[idx]["slide_id"],
            slide=self.slide_tiles.slide_path.stem,
            x=self.slide_tiles.tiles.iloc[idx]["x"],
            y=self.slide_tiles.tiles.iloc[idx]["y"],
        )


class BaseDataset(MetaTiledSlides[T], ABC):
    def __init__(
        self,
        uris: Iterable[str],
        sample_constructor: type[SlideTiles[T]],
        slide_ids: list[str] | None = None,
        transforms: Any | None = None,
    ) -> None:
        self.transforms = transforms
        self.sample_constructor = sample_constructor
        self.slide_ids = slide_ids
        super().__init__(uris=uris)

    def _prepare_data_hook(self) -> None:
        pass

    def generate_datasets(self) -> Iterable[Dataset[T]]:
        if self.slide_ids is not None:
            self.slides = self.slides[self.slides["path"].isin(self.slide_ids)]

        self.tiles["cancer"] = self.tiles["metastazis"] > 0

        self.tiles["gb-kind"] = "normal"
        self.tiles.loc[self.tiles["cancer"], "gb-kind"] = "cancer"
        self.tiles.loc[
            ~self.tiles["cancer"] & (self.tiles["brownish"] > 0), "gb-kind"
        ] = "brownish"
        # Make sure it is ordered by cateogry when using order by
        self.tiles["gb-kind"] = (
            self.tiles["gb-kind"]
            .astype("category")
            .cat.set_categories(["cancer", "brownish", "normal"], ordered=True)
        )

        self._prepare_data_hook()

        return (
            self.sample_constructor(
                slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )

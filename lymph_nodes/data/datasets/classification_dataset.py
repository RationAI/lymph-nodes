from collections.abc import Iterable
from typing import Any

import torch

from lymph_nodes.data.datasets.base_dataset import BaseDataset, SlideTiles
from lymph_nodes.typing import PredictSample, Sample


class _ClassificationSlideTiles(SlideTiles[Sample]):
    def __getitem__(self, idx: int) -> Sample:
        np_image = self.slide_tiles[idx]
        metadata = self._get_metadata(idx)

        if self.transforms is not None:
            np_image = self.transforms(image=np_image)["image"]

        image = self.to_tensor(image=np_image)["image"]

        label = torch.tensor([self.slide_tiles.tiles.iloc[idx]["cancer"]]).float()
        return image, label, metadata


class _ClassificationSlideTilesPred(SlideTiles[PredictSample]):
    def __getitem__(self, idx: int) -> PredictSample:
        np_image = self.slide_tiles[idx]
        metadata = self._get_metadata(idx)

        if self.transforms is not None:
            np_image = self.transforms(image=np_image)["image"]

        image = self.to_tensor(image=np_image)["image"]

        return image, metadata


class ClassificationDataset(BaseDataset[Sample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: Any | None = None,
    ) -> None:
        super().__init__(
            uris=uris,
            sample_constructor=_ClassificationSlideTiles,
            transforms=transforms,
        )


class ClassificationPredictDataset(BaseDataset[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str],
        transforms: Any | None = None,
    ) -> None:
        super().__init__(
            uris=uris,
            sample_constructor=_ClassificationSlideTilesPred,
            transforms=transforms,
        )

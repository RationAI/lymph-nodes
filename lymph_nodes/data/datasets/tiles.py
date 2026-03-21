from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from datasets import load_dataset
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide
from torch.utils.data import ConcatDataset, Dataset

from lymph_nodes.typing import TileMetadata, TilesPredictSample, TilesSample


class OpenSlideTilesDataset(Dataset[NDArray[np.uint8]]):
    def __init__(
        self,
        slide_path: str | Path,
        level: int | str,
        tile_extent_x: int | str,
        tile_extent_y: int | str,
        tiles: HFDataset,
    ) -> None:
        super().__init__()
        self.slide_path = Path(slide_path)
        self.level = level
        self.tile_extent_x = tile_extent_x
        self.tile_extent_y = tile_extent_y
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        tile = self.tiles[idx]
        level = self._get_from_tile(tile, self.level)
        extent_x = self._get_from_tile(tile, self.tile_extent_x)
        extent_y = self._get_from_tile(tile, self.tile_extent_y)

        with OpenSlide(self.slide_path) as slide:
            return slide.read_tile(tile["x"], tile["y"], extent_x, extent_y, level)

    @staticmethod
    def _get_from_tile(tile: dict[str, Any], key: int | str) -> int:
        return tile[key] if isinstance(key, str) else key


class _Tiles[T: TilesPredictSample | TilesSample](Dataset[T]):
    def __init__(
        self,
        slide_metadata: dict[str, Any],
        tiles: HFDataset,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata["path"],
            level=slide_metadata.get("level", "level"),
            tile_extent_x=slide_metadata.get("tile_extent_x", "tile_extent_x"),
            tile_extent_y=slide_metadata.get("tile_extent_y", "tile_extent_y"),
            tiles=tiles,
        )
        self.slide_metadata = slide_metadata
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, index: int) -> TilesPredictSample:  # | TilesSample
        image = self.slide_tiles[index]
        tile_row = self.slide_tiles.tiles[index]
        metadata: TileMetadata = {
            "slide_id": self.slide_tiles.slide_path.stem,
            "x": int(tile_row["x"]),
            "y": int(tile_row["y"]),
        }

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]

        return image, metadata


class TilesPredict(ConcatDataset[TilesPredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        transforms: TransformType | None = None,
    ) -> None:
        if isinstance(uris, str):
            uris = [uris]

        self._slide_datasets: list[_Tiles] = []

        for uri in uris:
            artifact_path = Path(download_artifacts(artifact_uri=uri))
            slides = load_dataset(
                "parquet",
                data_files=str(artifact_path / "slides.parquet"),
                split="train",
            )
            tiles = load_dataset(
                "parquet",
                data_files=str(artifact_path / "tiles.parquet"),
                split="train",
            )

            if "tile_x" in tiles.column_names:
                tiles = tiles.rename_columns({"tile_x": "x", "tile_y": "y"})

            for slide in slides:
                self._slide_datasets.append(
                    _Tiles(
                        slide_metadata=slide,
                        tiles=tiles.filter(
                            lambda row, sid=slide["id"]: row["slide_id"] == sid,
                            keep_in_memory=False,
                        ),
                        transforms=transforms,
                    )
                )

        super().__init__(self._slide_datasets)

    def generate_datasets(self) -> Iterable[_Tiles[TilesPredictSample]]:
        return iter(self._slide_datasets)

from collections.abc import Iterable
from pathlib import Path

from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from mlflow.artifacts import download_artifacts
from rationai.mlkit.data.datasets import OpenSlideTilesDataset
from torch.utils.data import ConcatDataset, Dataset

from lymph_nodes.typing import TileMetadata, TilesPredictSample


class _Tiles(Dataset[TilesPredictSample]):
    def __init__(
        self,
        slide_tiles: OpenSlideTilesDataset,
        slide_id: str,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = slide_tiles
        self._slide_id = slide_id
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, index: int) -> TilesPredictSample:
        image = self.slide_tiles[index]
        tile_row = self.slide_tiles.tiles[index]
        metadata: TileMetadata = {
            "slide_id": self._slide_id,
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
        self._artifact_paths: list[Path] = []

        for uri in uris:
            artifact_path = Path(download_artifacts(artifact_uri=uri))
            self._artifact_paths.append(artifact_path)
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
                slide_tiles_hf = tiles.filter(
                    lambda row, sid=slide["id"]: row["slide_id"] == sid,
                    keep_in_memory=False,
                )

                slide_tiles = OpenSlideTilesDataset(
                    slide_path=slide["path"],
                    level=slide.get("level", 0),
                    tile_extent_x=slide.get("tile_extent_x", 224),
                    tile_extent_y=slide.get("tile_extent_y", 224),
                    tiles=slide_tiles_hf,
                )
                self._slide_datasets.append(
                    _Tiles(slide_tiles, slide_id=slide["id"], transforms=transforms)
                )

        super().__init__(self._slide_datasets)

    @property
    def artifact_paths(self) -> list[Path]:
        return self._artifact_paths

    def generate_datasets(self) -> Iterable[_Tiles]:
        return iter(self._slide_datasets)

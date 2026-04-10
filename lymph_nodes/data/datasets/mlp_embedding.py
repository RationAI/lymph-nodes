from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset


if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)


def _label_from_filename(stem: str) -> int:
    """Helper to extract slide-level labels from slide IDs."""
    if stem.endswith("-1"):
        return 1
    if stem.endswith("-0"):
        return 0
    return 1


class SlideEmbeddingDataset(Dataset):
    """The leaf dataset representing a single slide."""

    def __init__(self, tiles: HFDataset, name: str, label: int) -> None:
        self._tiles = tiles.with_format("numpy")
        self.name = name
        self.label = label

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self._tiles[idx]

        embedding = torch.from_numpy(row["embedding"].copy())
        label = torch.tensor(float(row["metastazis"]), dtype=torch.float32)

        metadata = {"x": row["x"], "y": row["y"], "slide_id": self.name}

        return embedding, label, metadata


class MLPEmbeddingDataset(MetaTiledSlides):
    def __init__(self, paths: list[str], uris: list[str] | None = None) -> None:
        # This calls generate_datasets internally
        super().__init__(paths=paths, uris=uris)

    def generate_datasets(self) -> Iterable[Dataset]:
        for slide in self.slides:
            slide_id = slide["id"]
            slide_tiles_view = self.filter_tiles_by_slide(slide_id)

            if len(slide_tiles_view) > 0:
                yield SlideEmbeddingDataset(
                    tiles=slide_tiles_view,
                    name=slide_id,
                    label=_label_from_filename(slide_id),
                )

    @property
    def labels(self) -> np.ndarray:
        """Required for WeightedRandomSampler."""
        return np.array(self.tiles["metastazis"], dtype=np.int8)

    @property
    def groups(self) -> np.ndarray:
        """Required for StratifiedGroupKFold."""
        return np.array(self.tiles["slide_id"])

    @property
    def slides(self) -> Any:
        if hasattr(self, "datasets") and self.datasets:
            return [{"name": ds.name, "label": ds.label} for ds in self.datasets]

        return self.__dict__.get("slides")

    @slides.setter
    def slides(self, value: Any) -> None:
        """Allows the parent class to set self.slides during init."""
        self.__dict__["slides"] = value

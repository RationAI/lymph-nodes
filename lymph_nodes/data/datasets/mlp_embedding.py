from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets.meta_tiled_slides import MetaTiledSlides
from torch.utils.data import Dataset


class SlideEmbeddingDataset(Dataset):
    """Per-slide dataset for pre-computed tile embeddings.

    Wraps a HuggingFace Dataset subset containing tiles for a single slide.
    Data is lazily converted to tensors in ``__getitem__`` to keep memory
    usage low via Arrow memory-mapping.
    """

    def __init__(self, tiles: HFDataset) -> None:
        self._tiles = tiles

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._tiles[idx]
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        label = torch.tensor(row["metastazis"], dtype=torch.float32)
        return embedding, label


class MLPEmbeddingDataset(MetaTiledSlides):
    """Concrete :class:`MetaTiledSlides` for MLP training on pre-computed embeddings.

    Loads ``slides.parquet`` and ``tiles.parquet`` via the parent class, then
    partitions tiles per slide using :meth:`filter_tiles_by_slide`.  All tile
    data stays memory-mapped (Apache Arrow) so 1.1 M tiles across 30 slides
    fit within a 16 GiB RAM budget.
    """

    def generate_datasets(self) -> Iterable[Dataset]:
        for slide in self.slides:
            slide_tiles = self.filter_tiles_by_slide(slide["id"])
            yield SlideEmbeddingDataset(tiles=slide_tiles)

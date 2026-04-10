from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets.meta_tiled_slides import MetaTiledSlides
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.tile_embeddings import _group_from_stem


if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)


class _SlideTiles(Dataset):
    """Per-tile dataset for a single slide, used internally by TilePatchDataset."""

    def __init__(self, label: int, name: str, group: str, tiles: HFDataset) -> None:
        self._label = label
        self._name = name
        self._group = group
        self._tiles = tiles

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        tile = self._tiles[idx]
        embedding = torch.tensor(tile["embedding"], dtype=torch.float32)
        label = torch.tensor(self._label, dtype=torch.float32)
        metadata = {
            "slide_name": self._name,
            "tile_x": tile["x"],
            "tile_y": tile["y"],
        }
        return embedding, label, metadata


class TilePatchDataset(MetaTiledSlides):
    """Per-tile dataset for patch-based MLP classification.

    Backed by :class:`MetaTiledSlides` which loads slides and tiles from
    parquet files (via local paths or MLflow artifact URIs).
    Each item is a single tile: ``(embedding, label, metadata)``.

    Expected parquet schema:
        ``slides.parquet``: ``slide_id``, ``name``, ``label``
        ``tiles.parquet``:  ``slide_id``, ``embedding``, ``x``, ``y``
    """

    def generate_datasets(self) -> Iterable[Dataset]:
        return (
            _SlideTiles(
                label=slide["label"],
                name=slide["name"],
                group=_group_from_stem(slide["name"]),
                tiles=self.filter_tiles_by_slide(slide["slide_id"]),
            )
            for slide in self.slides
        )

    @property
    def labels(self) -> list[int]:
        """Flat per-tile label list (compatible with StratifiedGroupKFold)."""
        return [ds._label for ds in self.datasets for _ in range(len(ds))]

    @property
    def groups(self) -> list[str]:
        """Flat per-tile group list (compatible with StratifiedGroupKFold)."""
        return [ds._group for ds in self.datasets for _ in range(len(ds))]

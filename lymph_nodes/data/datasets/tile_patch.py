from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets.meta_tiled_slides import MetaTiledSlides
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.tile_embeddings import (
    _group_from_stem,
    _label_from_filename,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)


class _SlideTiles(Dataset):
    """Per-tile dataset for a single slide's pre-computed embeddings."""

    def __init__(self, name: str, label: int, group: str, tiles: HFDataset) -> None:
        self._name = name
        self._label = label
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

    Subclasses :class:`MetaTiledSlides` which loads ``slides.parquet`` and
    ``tiles.parquet`` from local directories or MLflow artifact URIs.
    Each item is ``(embedding, label, metadata)``.

    Expected parquet schema:
        ``slides.parquet``: columns including ``id`` (slide identifier)
        ``tiles.parquet``:  ``slide_id``, ``embedding``, ``x``, ``y``
    """

    def __init__(self, **kwargs) -> None:
        log.info("TilePatchDataset: loading slides and tiles …")
        super().__init__(**kwargs)
        # Replace HFDataset self.slides with list-of-dicts expected by
        # DatasetSubset and _log_split.
        self.slides = [  # type: ignore[assignment]
            {"name": ds._name, "label": ds._label} for ds in self.datasets
        ]
        n_tiles = sum(len(ds) for ds in self.datasets)
        n_pos = sum(1 for ds in self.datasets if ds._label == 1)
        n_neg = len(self.datasets) - n_pos
        log.info(
            "TilePatchDataset ready: %d tiles from %d slides (%d+ / %d-)",
            n_tiles,
            len(self.datasets),
            n_pos,
            n_neg,
        )
        for ds in self.datasets:
            marker = "+" if ds._label == 1 else "-"
            log.info("  [%s] %-40s  %d tiles", marker, ds._name, len(ds))

    def generate_datasets(self) -> Iterable[Dataset]:
        return (
            _SlideTiles(
                name=slide["id"],
                label=_label_from_filename(slide["id"]),
                group=_group_from_stem(slide["id"]),
                tiles=self.filter_tiles_by_slide(slide["id"]),
            )
            for slide in self.slides
        )

    @cached_property
    def labels(self) -> list[int]:
        """Flat per-tile label list (for StratifiedGroupKFold / WeightedRandomSampler)."""
        return [ds._label for ds in self.datasets for _ in range(len(ds))]

    @cached_property
    def groups(self) -> list[str]:
        """Flat per-tile group list (for StratifiedGroupKFold)."""
        return [ds._group for ds in self.datasets for _ in range(len(ds))]

    @cached_property
    def _tile_slide_names(self) -> list[str]:
        """Flat per-tile slide name list (for DatasetSubset slide deduplication)."""
        return [ds._name for ds in self.datasets for _ in range(len(ds))]

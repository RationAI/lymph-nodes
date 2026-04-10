from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from datasets.table import InMemoryTable
from mlflow.artifacts import download_artifacts
from rationai.mlkit.data.datasets.meta_tiled_slides import MetaTiledSlides
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import ConcatDataset, Dataset

from lymph_nodes.data.datasets.tile_embeddings import (
    _group_from_stem,
    _label_from_filename,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)


class SlideEmbeddingDataset(Dataset):
    """Worker Dataset: Returns (embedding, label, metadata).

    Holds a shared reference to the full memory-mapped tiles HFDataset and a
    list of row indices for this slide.  Each __getitem__ reads exactly one row
    lazily, so no slide's embeddings are materialised into RAM upfront.
    """

    def __init__(
        self,
        *,
        name: str,
        label: int,
        group: str,
        tiles: HFDataset,
        indices: list[int],
    ) -> None:
        self._name = name
        self._label = label
        self._group = group
        self._tiles = tiles  # shared reference — never copied
        self._indices = indices  # per-slide row pointers into _tiles

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self._tiles[self._indices[idx]]  # single lazy row read
        # 1. Feature Vector
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        # 2. Target Label (Patch-level label from parquet)
        label = torch.tensor(row["metastazis"], dtype=torch.float32)
        # 3. Metadata for heatmaps
        metadata = {"x": row["x"], "y": row["y"], "slide_id": row["slide_id"]}

        return embedding, label, metadata


class _SlideSubset(ConcatDataset):
    """Memory-efficient subset of MLPEmbeddingDataset for selected slides.

    Instead of copying per-tile Python lists (~300 MB for labels alone),
    this wraps the selected ``SlideEmbeddingDataset`` objects in a new
    ``ConcatDataset`` and derives per-tile labels via numpy (~10 MB).
    """

    def __init__(
        self,
        parent: MLPEmbeddingDataset,
        slide_indices: Sequence[int],
    ) -> None:
        selected = [parent.datasets[i] for i in slide_indices]
        super().__init__(selected)
        self.slides = [parent.slides[i] for i in slide_indices]

    @cached_property
    def labels(self) -> np.ndarray:
        """Per-tile labels as a compact numpy array (int8, ~10 MB for 10 M tiles)."""
        slide_labels = np.array([ds._label for ds in self.datasets], dtype=np.int8)
        slide_lengths = np.array([len(ds) for ds in self.datasets])
        return np.repeat(slide_labels, slide_lengths)

    def sample_weights(self, pos_weight: float = 1.0) -> torch.Tensor:
        """Per-tile sampling weights with slide-balanced class weighting.

        Each slide contributes equally within its class so that large slides
        don't dominate.  ``pos_weight`` controls the ratio between total
        positive and total negative weight:

        * ``1.0`` — balanced (each class sampled ~50 % of the time)
        * ``< 1`` — partial upweight (e.g. ``0.5`` → pos sampled ~33 %)
        * ``> 1`` — oversample positives even further
        """
        n_pos_slides = sum(1 for ds in self.datasets if ds._label == 1)
        n_neg_slides = len(self.datasets) - n_pos_slides

        weights = torch.empty(len(self), dtype=torch.float64)
        offset = 0
        for ds in self.datasets:
            n = len(ds)
            if ds._label == 1 and n_pos_slides > 0:
                # Equal contribution per positive slide, scaled by pos_weight.
                w = pos_weight / (n_pos_slides * n)
            elif n_neg_slides > 0:
                w = 1.0 / (n_neg_slides * n)
            else:
                w = 1.0
            weights[offset : offset + n] = w
            offset += n
        return weights


class MLPEmbeddingDataset(MetaTiledSlides):
    """Manager Dataset: Handles artifact retrieval and lazy-loading schema fixes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Replace HFDataset rows with lightweight dicts expected by _log_split.
        self.slides = [
            {"id": ds._name, "name": ds._name, "label": ds._label, "group": ds._group}
            for ds in self.datasets
        ]

    def generate_datasets(self) -> Iterable[Dataset]:
        """Yields datasets using the 'id' column from the initial slides table."""
        print(f"📦 Generating datasets for {len(self.slides)} slides...")

        # Capture shared references so every sub-dataset points to the same
        # memory-mapped HFDataset instead of creating separate .select() copies.
        tiles = self.tiles
        index = self._slide_id_to_indices

        return (
            SlideEmbeddingDataset(
                name=slide["id"],
                label=_label_from_filename(slide["id"]),
                group=_group_from_stem(slide["id"]),
                tiles=tiles,
                indices=index.get(slide["id"], []),
            )
            for slide in self.slides
            if index.get(slide["id"])
        )

    def kfold_split(
        self,
        n_splits: int,
        k: int,
        *,
        random_state: int = 42,
        pos_weight: float = 0.5,
    ) -> tuple[_SlideSubset, _SlideSubset]:
        """Stratified group K-fold split at **slide level** (not tile level).

        Avoids materialising 13 M+ element Python lists for labels/groups
        by splitting over the ~366 slides and returning lightweight
        ``_SlideSubset`` wrappers.

        Args:
            n_splits: Total number of folds.
            k: 1-based index of the fold to use as validation.
            random_state: Random seed for reproducible splits.
            pos_weight: Upweight factor for positive slides in the sampler.
                1.0 = fully balanced, 0.5 = partial balance.
        """
        slide_labels = [ds._label for ds in self.datasets]
        slide_groups = [ds._group for ds in self.datasets]
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = list(sgkf.split(range(len(self.datasets)), slide_labels, slide_groups))
        train_idx, val_idx = splits[k - 1]
        train_subset = _SlideSubset(self, train_idx)
        val_subset = _SlideSubset(self, val_idx)
        train_subset._pos_weight = pos_weight
        val_subset._pos_weight = 1.0  # always balanced for val
        return train_subset, val_subset

    @staticmethod
    def load_slides_and_tiles(
        paths: Iterable[str | Path], uris: Iterable[str]
    ) -> tuple[HFDataset, HFDataset]:
        """Custom loader for schema promotion and strict column filtering."""
        print("🔍 Locating artifacts...")
        with ThreadPoolExecutor() as executor:
            artifact_paths = list(
                executor.map(lambda uri: download_artifacts(artifact_uri=uri), uris)
            )

        search_dirs = [Path(p) for p in (*paths, *artifact_paths)]
        slide_files = [
            p / "slides.parquet" for p in search_dirs if (p / "slides.parquet").exists()
        ]
        tile_files = [
            str(p / "tiles.parquet")
            for p in search_dirs
            if (p / "tiles.parquet").exists()
        ]

        if not slide_files or not tile_files:
            return HFDataset.from_dict({}), HFDataset.from_dict({})

        # Slides: PyArrow concat (safe for null/string conflicts)
        slides_table = pa.concat_tables(
            [pq.read_table(str(f)) for f in slide_files],
            promote_options="default",
        )
        slides_ds = HFDataset(InMemoryTable(slides_table))

        # Tiles: Lazy HuggingFace mapping
        TILE_COLS = ["x", "y", "embedding", "slide_id", "metastazis"]
        tiles_ds = cast(
            "HFDataset",
            load_dataset(
                "parquet", split="train", data_files=tile_files, columns=TILE_COLS
            ),
        )

        return slides_ds, tiles_ds

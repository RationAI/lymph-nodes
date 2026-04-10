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

log = logging.getLogger(__name__)


class SlideEmbeddingDataset(Dataset):
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
        self._tiles = tiles
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self._tiles[self._indices[idx]]
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        label = torch.tensor(row["metastazis"], dtype=torch.float32)
        metadata = {"x": row["x"], "y": row["y"], "slide_id": row["slide_id"]}
        return embedding, label, metadata


class _SlideSubset(ConcatDataset):
    def __init__(
        self, parent: MLPEmbeddingDataset, slide_indices: Sequence[int]
    ) -> None:
        selected = [parent.datasets[i] for i in slide_indices]
        super().__init__(selected)
        self.slides = [parent.slides[i] for i in slide_indices]

    @cached_property
    def labels(self) -> np.ndarray:
        slide_labels = np.array([ds._label for ds in self.datasets], dtype=np.int8)
        slide_lengths = np.array([len(ds) for ds in self.datasets])
        return np.repeat(slide_labels, slide_lengths)

    def sample_weights(self, pos_weight: float = 1.0) -> torch.Tensor:
        n_pos_slides = sum(1 for ds in self.datasets if ds._label == 1)
        n_neg_slides = len(self.datasets) - n_pos_slides
        weights = torch.empty(len(self), dtype=torch.float64)
        offset = 0
        for ds in self.datasets:
            n = len(ds)
            if ds._label == 1 and n_pos_slides > 0:
                w = pos_weight / (n_pos_slides * n)
            elif n_neg_slides > 0:
                w = 1.0 / (n_neg_slides * n)
            else:
                w = 1.0
            weights[offset : offset + n] = w
            offset += n
        return weights


class MLPEmbeddingDataset(MetaTiledSlides):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.slides = [
            {"id": ds._name, "name": ds._name, "label": ds._label, "group": ds._group}
            for ds in self.datasets
        ]

    def generate_datasets(self) -> Iterable[Dataset]:
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
        val_subset._pos_weight = 1.0
        return train_subset, val_subset

    @cached_property
    def labels(self) -> list[int]:
        return [ds._label for ds in self.datasets for _ in range(len(ds))]

    @cached_property
    def groups(self) -> list[str]:
        return [ds._group for ds in self.datasets for _ in range(len(ds))]

    @cached_property
    def _tile_slide_names(self) -> list[str]:
        return [ds._name for ds in self.datasets for _ in range(len(ds))]

    @staticmethod
    def load_slides_and_tiles(
        paths: Iterable[str | Path], uris: Iterable[str]
    ) -> tuple[HFDataset, HFDataset]:
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

        slides_table = pa.concat_tables(
            [pq.read_table(str(f)) for f in slide_files],
            promote_options="default",
        )
        slides_ds = HFDataset(InMemoryTable(slides_table))

        TILE_COLS = ["x", "y", "embedding", "slide_id", "metastazis"]
        tiles_ds = cast(
            "HFDataset",
            load_dataset(
                "parquet", split="train", data_files=tile_files, columns=TILE_COLS
            ),
        )

        return slides_ds, tiles_ds

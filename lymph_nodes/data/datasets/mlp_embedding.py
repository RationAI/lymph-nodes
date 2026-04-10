from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from datasets.table import InMemoryTable
from mlflow.artifacts import download_artifacts
from rationai.mlkit.data.datasets.meta_tiled_slides import MetaTiledSlides
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.tile_embeddings import (
    _group_from_stem,
    _label_from_filename,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class SlideEmbeddingDataset(Dataset):
    """Worker Dataset: Returns (embedding, label, metadata)."""

    def __init__(self, tiles: HFDataset, *, name: str, label: int, group: str) -> None:
        self._tiles = tiles
        self._name = name
        self._label = label
        self._group = group

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self._tiles[idx]
        # 1. Feature Vector
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        # 2. Target Label (Patch-level label from parquet)
        label = torch.tensor(row["metastazis"], dtype=torch.float32)
        # 3. Metadata for heatmaps
        metadata = {"x": row["x"], "y": row["y"], "slide_id": row["slide_id"]}

        return embedding, label, metadata


class MLPEmbeddingDataset(MetaTiledSlides):
    """Manager Dataset: Handles artifact retrieval and lazy-loading schema fixes."""

    def __init__(self, **kwargs: Any) -> None:
        # super().__init__ triggers load_slides_and_tiles and generate_datasets
        super().__init__(**kwargs)

        # After generation, we update self.slides metadata without losing 'id'
        # We use .get() to be safe if generate_datasets is called multiple times
        self.slides = [
            {"id": ds._name, "name": ds._name, "label": ds._label, "group": ds._group}
            for ds in self.datasets
        ]

    def generate_datasets(self) -> Iterable[Dataset]:
        """Yields datasets using the 'id' column from the initial slides table."""
        print(f"📦 Generating datasets for {len(self.slides)} slides...")

        # At first call, self.slides is the HFDataset from load_slides_and_tiles
        for slide in self.slides:
            # We must use 'id' because that's what's in the slides.parquet
            slide_id = slide["id"]
            slide_tiles = self.filter_tiles_by_slide(slide_id)

            if len(slide_tiles) > 0:
                yield SlideEmbeddingDataset(
                    tiles=slide_tiles,
                    name=slide_id,
                    label=_label_from_filename(slide_id),
                    group=_group_from_stem(slide_id),
                )

    @cached_property
    def labels(self) -> list[int]:
        """Flat per-tile label list for Stratified splitting."""
        # Using a list comprehension over 13.6M items is okay (~100MB RAM)
        print("🧮 Calculating global label list for stratification...")
        return [ds._label for ds in self.datasets for _ in range(len(ds))]

    @cached_property
    def groups(self) -> list[str]:
        """Flat per-tile group list for GroupKFold splitting."""
        print("👥 Calculating global group list for splitting...")
        return [ds._group for ds in self.datasets for _ in range(len(ds))]

    @cached_property
    def _tile_slide_names(self) -> list[str]:
        """Flat per-tile slide name list for deduplication."""
        return [ds._name for ds in self.datasets for _ in range(len(ds))]

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

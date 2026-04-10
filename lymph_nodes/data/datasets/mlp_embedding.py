from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
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


if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class SlideEmbeddingDataset(Dataset):
    """Worker Dataset: Returns (embedding, label, metadata).

    Matches PatchMLP.training_step expectations.
    """

    def __init__(self, tiles: HFDataset) -> None:
        self._tiles = tiles

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self._tiles[idx]

        # 1. Feature Vector (2560-dim)
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)

        # 2. Target Label
        label = torch.tensor(row["metastazis"], dtype=torch.float32)

        # 3. Metadata for heatmap generation and training_step unpacking
        metadata = {"x": row["x"], "y": row["y"], "slide_id": row["slide_id"]}

        return embedding, label, metadata


class MLPEmbeddingDataset(MetaTiledSlides):
    """Manager Dataset: Handles artifact retrieval and lazy-loading schema fixes."""

    def generate_datasets(self) -> Iterable[Dataset]:
        print(f"📦 Generating datasets for {len(self.slides)} slides...")
        for slide in self.slides:
            slide_tiles = self.filter_tiles_by_slide(slide["id"])
            if len(slide_tiles) > 0:
                yield SlideEmbeddingDataset(tiles=slide_tiles)

    @staticmethod
    def load_slides_and_tiles(
        paths: Iterable[str | Path], uris: Iterable[str]
    ) -> tuple[HFDataset, HFDataset]:
        """Custom loader for schema promotion and strict column filtering.

        Handles:
        1. PyArrow concat with schema promotion for slides.
        2. Strict column filtering for tiles to avoid CastErrors.
        """
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
            print("⚠️ No Parquet files found in the specified paths/URIs.")
            return HFDataset.from_dict({}), HFDataset.from_dict({})

        # --- SLIDES PROCESSING ---
        print(f"📊 Loading {len(slide_files)} slide metadata files...")
        slides_table = pa.concat_tables(
            [pq.read_table(str(f)) for f in slide_files],
            promote_options="default",
        )
        slides_ds = HFDataset(InMemoryTable(slides_table))
        print(f"✅ Slides Loaded. Total slides: {len(slides_ds)}")

        # --- TILES PROCESSING ---
        print(f"🏗️ Loading {len(tile_files)} tile embedding files (Lazy Mode)...")
        TILE_COLS = ["x", "y", "embedding", "slide_id", "metastazis"]

        tiles_ds = cast(
            "HFDataset",
            load_dataset(
                path="parquet", split="train", data_files=tile_files, columns=TILE_COLS
            ),
        )
        print(f"✅ Tiles Mapped. Total tiles: {len(tiles_ds):,}")

        return slides_ds, tiles_ds

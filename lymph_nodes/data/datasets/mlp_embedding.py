from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from collections.abc import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from datasets.table import InMemoryTable
from mlflow.artifacts import download_artifacts
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

    @staticmethod
    def load_slides_and_tiles(
        paths: Iterable[str | Path], uris: Iterable[str]
    ) -> tuple[HFDataset, HFDataset]:
        """Load slides/tiles with schema promotion to handle null-type columns.

        The base-class implementation uses ``load_dataset`` for both slides and
        tiles, which fails when multiple parquet files have incompatible
        null-typed columns (e.g. ``cytokeratin_mask_path``).  This override
        uses PyArrow ``concat_tables`` with schema promotion for the small
        slides table, and ``load_dataset`` (memory-mapped) for the large tiles.
        """
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

        # Slides: small table — use PyArrow concat with schema promotion
        # to handle null-typed columns that differ across sources.
        slides_table = pa.concat_tables(
            [pq.read_table(str(f)) for f in slide_files],
            promote_options="default",
        )
        slides_ds = HFDataset(InMemoryTable(slides_table))

        # Tiles: large table — load lazily via HuggingFace memory-mapping.
        tiles_ds = cast(
            "HFDataset",
            load_dataset(path="parquet", split="train", data_files=tile_files),
        )

        return slides_ds, tiles_ds

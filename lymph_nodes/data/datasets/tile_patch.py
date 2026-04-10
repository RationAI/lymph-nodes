from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
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

from lymph_nodes.data.datasets.tile_embeddings import (
    _group_from_stem,
    _label_from_filename,
)


log = logging.getLogger(__name__)


class _SlideTiles(Dataset):
    """Per-tile dataset for a single slide's pre-computed embeddings.

    Holds a reference to the shared, memory-mapped full tiles HFDataset and a
    list of row indices for this slide.  Access is lazy: each __getitem__ reads
    exactly one row from the Arrow mmap cache, so no slide's embeddings are
    materialised into RAM upfront.
    """

    def __init__(
        self,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        tile = self._tiles[self._indices[idx]]  # single lazy row read
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
        # Capture both references eagerly so the generator closure is safe.
        tiles = self.tiles
        index = self._slide_id_to_indices
        return (
            _SlideTiles(
                name=slide["id"],
                label=_label_from_filename(slide["id"]),
                group=_group_from_stem(slide["id"]),
                tiles=tiles,
                indices=index.get(slide["id"], []),
            )
            for slide in self.slides
        )

    @staticmethod
    def load_slides_and_tiles(
        paths: Iterable[str | Path], uris: Iterable[str]
    ) -> tuple[HFDataset, HFDataset]:
        """Load slides/tiles with schema promotion to handle null-type columns."""
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

        # Slides: only ~1 row per slide — use PyArrow concat with schema promotion
        # to handle null-typed columns (e.g. cytokeratin_mask_path) that differ
        # across sources.
        slides_table = pa.concat_tables(
            [pq.read_table(str(f)) for f in slide_files],
            promote_options="default",
        )
        slides_ds = HFDataset(InMemoryTable(slides_table))

        # Tiles: millions of rows with large embeddings — load lazily via HuggingFace
        # memory-mapping so embeddings are only read from disk on access.
        tiles_ds = cast(
            "HFDataset",
            load_dataset(path="parquet", split="train", data_files=tile_files),
        )

        return slides_ds, tiles_ds

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

import logging
from pathlib import Path

import mlflow.artifacts
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.tile_embeddings import (
    _group_from_stem,
    _label_from_filename,
)


log = logging.getLogger(__name__)


class TilePatchDataset(Dataset):
    """Per-tile dataset for patch-based MLP classification.

    Uses HuggingFace ``datasets`` for lazy (memory-mapped) Parquet loading so
    that embeddings are read from disk on demand instead of being materialized
    in RAM all at once.
    """

    def __init__(
        self,
        embeddings_uri: str | list[str],
        include_slides: list[str] | None = None,
    ) -> None:
        uris = [embeddings_uri] if isinstance(embeddings_uri, str) else embeddings_uri
        parquet_files: list[Path] = []
        for uri in uris:
            embeddings_dir = Path(mlflow.artifacts.download_artifacts(uri))
            parquet_files.extend(embeddings_dir.rglob("*.parquet"))
        parquet_files = sorted(set(parquet_files))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in any of: {uris}")

        if include_slides is not None:
            include_set = set(include_slides)
            parquet_files = [pf for pf in parquet_files if pf.stem in include_set]
            if not parquet_files:
                raise FileNotFoundError(
                    "No parquet files matched include_slides in provided URIs"
                )

        tile_labels: list[int] = []
        tile_groups: list[str] = []
        tile_slide_names: list[str] = []
        unique_slides: list[dict] = []

        for pf in parquet_files:
            n_rows = pq.read_metadata(pf).num_rows
            slide_name = pf.stem
            label = _label_from_filename(slide_name)
            group = _group_from_stem(slide_name)

            unique_slides.append({"name": slide_name, "path": pf, "label": label})
            tile_labels.extend([label] * n_rows)
            tile_groups.extend([group] * n_rows)
            tile_slide_names.extend([slide_name] * n_rows)

        log.info(
            "Loading %d parquet files via HuggingFace datasets (lazy) …",
            len(parquet_files),
        )
        self._ds = load_dataset(
            "parquet",
            data_files=[str(pf) for pf in parquet_files],
            split="train",
        )

        self.labels: list[int] = tile_labels
        self.groups: list[str] = tile_groups
        self.slides: list[dict] = unique_slides
        self._tile_slide_names: list[str] = tile_slide_names

        log.info(
            "TilePatchDataset ready: %d tiles from %d slides",
            len(self._ds),
            len(unique_slides),
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        row = self._ds[idx]
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        metadata = {
            "slide_name": self._tile_slide_names[idx],
            "tile_x": row.get("x", 0),
            "tile_y": row.get("y", 0),
        }
        return embedding, label, metadata

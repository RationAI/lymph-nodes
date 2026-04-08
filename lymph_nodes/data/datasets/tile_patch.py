from pathlib import Path

import mlflow.artifacts
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.tile_embeddings import (
    _group_from_stem,
    _label_from_filename,
)


class TilePatchDataset(Dataset):
    """Per-tile dataset for patch-based MLP classification."""

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
        tile_index: list[tuple[Path, int, int, int]] = []
        unique_slides: list[dict] = []

        for pf in parquet_files:
            label = _label_from_filename(pf.stem)
            group = _group_from_stem(pf.stem)
            schema = pq.read_schema(pf)
            n_rows = pq.read_metadata(pf).num_rows
            col_names = schema.names
            coord_cols = [c for c in ("x", "y") if c in col_names]
            if coord_cols:
                coord_df = pd.read_parquet(pf, columns=coord_cols)
                xs = (
                    coord_df["x"].to_numpy(dtype=np.int64)
                    if "x" in col_names
                    else np.zeros(n_rows, dtype=np.int64)
                )
                ys = (
                    coord_df["y"].to_numpy(dtype=np.int64)
                    if "y" in col_names
                    else np.zeros(n_rows, dtype=np.int64)
                )
            else:
                xs = np.zeros(n_rows, dtype=np.int64)
                ys = np.zeros(n_rows, dtype=np.int64)
            for i in range(len(xs)):
                tile_index.append((pf, i, int(xs[i]), int(ys[i])))
                tile_labels.append(label)
                tile_groups.append(group)
            unique_slides.append({"name": pf.stem, "path": pf, "label": label})

        self._tile_index = tile_index
        self.labels: list[int] = tile_labels
        self.groups: list[str] = tile_groups
        self.slides: list[dict] = unique_slides
        # Per-process cache: populated lazily by workers, each worker builds its own copy.
        self._emb_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._tile_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        pf, row, x, y = self._tile_index[idx]
        if pf not in self._emb_cache:
            self._emb_cache[pf] = np.stack(
                pd.read_parquet(pf, columns=["embedding"])["embedding"].tolist()
            ).astype(np.float32)
        embedding = torch.from_numpy(self._emb_cache[pf][row].copy())
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        metadata = {
            "slide_name": pf.stem,
            "slide_path": pf,
            "tile_x": x,
            "tile_y": y,
        }
        return embedding, label, metadata

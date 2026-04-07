from pathlib import Path

import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.tile_embeddings import (
    _group_from_stem,
    _label_from_filename,
)


class TilePatchDataset(Dataset):
    """Per-tile dataset for patch-based MLP classification.

    Each sample is a single tile embedding with its slide-level label propagated
    as the tile label.  All embeddings are loaded into memory during construction.
    For very large datasets (> ~10k tiles x 2560-dim) this may consume several
    gigabytes; reduce scope via ``include_slides`` if needed.

    Args:
        embeddings_uri:
            MLflow artifact URI pointing to a directory of per-slide ``.parquet``
            files.  Files must follow the ``<name>-<0|1>.parquet`` naming convention
            used throughout the project.
        include_slides:
            Optional whitelist of parquet file stems.  Only slides whose stem
            appears in this list are loaded.
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
                    f"No parquet files matched include_slides in {embeddings_dir}"
                )

        slide_infos: list[tuple[Path, int, str]] = []  # (path, label, group)
        for pf in parquet_files:
            slide_infos.append(
                (pf, _label_from_filename(pf.stem), _group_from_stem(pf.stem))
            )

        # Load all tile embeddings into a single contiguous numpy array.
        # This avoids opening parquet files on every __getitem__ call.
        all_embs: list[np.ndarray] = []
        tile_labels: list[int] = []
        tile_groups: list[str] = []
        tile_meta: list[
            tuple[str, Path, int, int]
        ] = []  # (slide_name, slide_path, x, y)
        unique_slides: list[dict] = []

        for pf, label, group in slide_infos:
            df = pd.read_parquet(pf)
            embs = np.stack(df["embedding"].tolist()).astype(np.float32)
            n = len(embs)
            xs = (
                df["x"].to_numpy(dtype=np.int64)
                if "x" in df.columns
                else np.zeros(n, dtype=np.int64)
            )
            ys = (
                df["y"].to_numpy(dtype=np.int64)
                if "y" in df.columns
                else np.zeros(n, dtype=np.int64)
            )

            all_embs.append(embs)
            tile_labels.extend([label] * n)
            tile_groups.extend([group] * n)
            for i in range(n):
                tile_meta.append((pf.stem, pf, int(xs[i]), int(ys[i])))

            unique_slides.append({"name": pf.stem, "path": pf, "label": label})

        self._embeddings: np.ndarray = (
            np.concatenate(all_embs, axis=0)
            if all_embs
            else np.empty((0, 0), dtype=np.float32)
        )
        self._tile_meta = tile_meta

        # Public attributes expected by DataModule / _log_split / _weighted_sampler.
        self.labels: list[int] = tile_labels
        self.groups: list[str] = tile_groups
        self.slides: list[dict] = unique_slides  # unique slides — used by _log_split

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tile_meta)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        embedding = torch.from_numpy(self._embeddings[idx].copy())
        slide_name, slide_path, x, y = self._tile_meta[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        metadata = {
            "slide_name": slide_name,
            "slide_path": slide_path,
            "tile_x": x,
            "tile_y": y,
        }
        return embedding, label, metadata

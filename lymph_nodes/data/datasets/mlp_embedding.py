from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from mlflow.artifacts import download_artifacts
from torch.utils.data import ConcatDataset, Dataset

from lymph_nodes.data.datasets.tile_embeddings import (
    _label_from_filename,
)


log = logging.getLogger(__name__)


class SlideEmbeddingDataset(Dataset):
    def __init__(
        self, tiles: HFDataset, name: str, indices: list[int], label: int
    ) -> None:
        self._tiles = tiles
        self.name = name
        self.indices = indices
        self.label = label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self._tiles[self.indices[idx]]
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        label = torch.tensor(row["metastazis"], dtype=torch.float32)
        metadata = {"x": row["x"], "y": row["y"], "slide_id": row["slide_id"]}
        return embedding, label, metadata


class MLPEmbeddingDataset(ConcatDataset):
    def __init__(self, paths: list[str], uris: list[str] | None = None) -> None:
        uris = uris or []
        with ThreadPoolExecutor() as executor:
            artifact_paths = list(
                executor.map(lambda u: download_artifacts(artifact_uri=u), uris)
            )
        all_dirs = [Path(p) for p in (*paths, *artifact_paths)]

        tile_files = [
            str(p / "tiles.parquet") for p in all_dirs if (p / "tiles.parquet").exists()
        ]
        slide_files = [
            str(p / "slides.parquet")
            for p in all_dirs
            if (p / "slides.parquet").exists()
        ]

        slides_table = pa.concat_tables(
            [pq.read_table(f) for f in slide_files],
            promote_options="default",
        )
        self._slides_raw = slides_table.to_pandas().to_dict("records")

        self.tiles_ds = cast(
            "HFDataset",
            load_dataset(
                "parquet",
                data_files=tile_files,
                split="train",
                columns=["x", "y", "embedding", "slide_id", "metastazis"],
            ),
        )

        index_map: dict[str, list[int]] = defaultdict(list)
        for idx, sid in enumerate(self.tiles_ds["slide_id"]):
            index_map[sid].append(idx)

        datasets = [
            SlideEmbeddingDataset(
                tiles=self.tiles_ds,
                name=sid,
                indices=index_map[sid],
                label=_label_from_filename(sid),
            )
            for s in self._slides_raw
            if (sid := s["id"]) in index_map
        ]
        super().__init__(datasets)
        self.datasets = datasets

    @cached_property
    def labels(self) -> np.ndarray:
        return np.array(self.tiles_ds["metastazis"], dtype=np.int8)

    @cached_property
    def groups(self) -> np.ndarray:
        return np.array(self.tiles_ds["slide_id"])

    @property
    def slides(self) -> list[dict]:
        return [{"name": ds.name, "label": ds.label} for ds in self.datasets]

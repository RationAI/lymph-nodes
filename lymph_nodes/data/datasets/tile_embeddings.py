from pathlib import Path

import mlflow.artifacts
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lymph_nodes.typing import MetadataTileEmbeddings, TileEmbeddingsSample


def _label_from_filename(stem: str) -> int:
    """Extract binary label from the filename convention.

    Both MMCI and FNBrno slides encode label as the last character:
      - ends with '-1' → positive (metastasis)
      - ends with '-0' → negative
    """
    if stem.endswith("-1"):
        return 1
    if stem.endswith("-0"):
        return 0
    raise ValueError(
        f"Cannot determine label from filename '{stem}'. "
        "Expected filename to end with '-0' (negative) or '-1' (positive)."
    )


class TileEmbeddings(Dataset[TileEmbeddingsSample]):
    """Dataset for pre-computed tile embeddings stored as per-slide parquet files.

    Each parquet file is expected to have columns: x, y, embedding
    where 'embedding' contains numpy arrays of shape (embedding_dim,).
    """

    def __init__(
        self,
        embeddings_uri: str,
        padding: bool = True,
    ) -> None:
        self.padding = padding

        embeddings_dir = Path(mlflow.artifacts.download_artifacts(embeddings_uri))

        parquet_files = sorted(embeddings_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {embeddings_dir}")

        self.slides: list[dict] = []
        for pf in parquet_files:
            label = _label_from_filename(pf.stem)
            self.slides.append(
                {
                    "name": pf.stem,
                    "path": pf,
                    "label": label,
                }
            )

        self.labels = [s["label"] for s in self.slides]

        if self.padding:
            self.max_tiles = max(len(pd.read_parquet(s["path"])) for s in self.slides)

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> TileEmbeddingsSample:
        slide = self.slides[idx]

        df = pd.read_parquet(slide["path"])
        embeddings = torch.from_numpy(np.stack(df["embedding"].tolist())).float()

        if self.padding:
            pad_amount = self.max_tiles - embeddings.shape[0]
            if pad_amount > 0:
                embeddings = F.pad(embeddings, (0, 0, 0, pad_amount), value=0.0)

        label = torch.tensor(slide["label"]).float()

        metadata = MetadataTileEmbeddings(
            slide_id=str(idx),
            slide_name=slide["name"],
            slide_path=slide["path"],
            tiles=df[["x", "y"]],
            x=torch.from_numpy(df["x"].to_numpy()),
            y=torch.from_numpy(df["y"].to_numpy()),
        )

        return embeddings, label, metadata

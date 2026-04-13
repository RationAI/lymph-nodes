import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import albumentations as A
import hydra
import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import login
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader

from lymph_nodes.data.datasets import TilesPredict


class FoundationModel(torch.nn.Module):
    def __init__(self, name: str, embed_dim: int) -> None:
        """Wrapper for a foundation model - forward and dimension differ depending on the model."""
        super().__init__()
        self.embed_dim = embed_dim


class Virchow2(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 2560)

        self.module = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)

        class_token = output[:, 0]
        patch_tokens = output[:, 5:]

        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


class ProvGigaPath(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1536)
        self.module = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        ).eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class UNI2(FoundationModel):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1536)

        self.module = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        ).eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


def load_dataset(uris: Iterable[str]) -> TilesPredict:
    """Load the dataset for tile embeddings.

    Assumes that the dataset has 224x224 RGB tiles.

    Args:
        uris (Iterable[str]): The URIs of the tiles.

    Returns:
        TilesPredict: The dataset object for tile embeddings.
    """
    return TilesPredict(
        uris,
        transforms=A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    )


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please set the HF_TOKEN environment variable."
        )
    login(token=hf_token)

    dest = Path(config.output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    output_tiles_path = dest / "tiles.parquet"
    output_slides_path = dest / "slides.parquet"
    if output_tiles_path.exists() and output_slides_path.exists():
        logger.log_artifacts(str(dest), artifact_path="embeddings")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tile_encoder: FoundationModel = hydra.utils.instantiate(config.tile_encoder)
    tile_encoder = tile_encoder.to(device)

    dataset = load_dataset(config.dataset.uris.values())

    # Read and merge the original tiles/slides parquets from all artifact URIs.
    original_tiles_parts: list[pd.DataFrame] = []
    original_slides_parts: list[pd.DataFrame] = []
    for artifact_path in dataset.artifact_paths:
        tiles_df = pd.read_parquet(artifact_path / "tiles.parquet")
        if "tile_x" in tiles_df.columns:
            tiles_df = tiles_df.rename(columns={"tile_x": "x", "tile_y": "y"})
        original_tiles_parts.append(tiles_df)
        original_slides_parts.append(pd.read_parquet(artifact_path / "slides.parquet"))

    original_tiles = pd.concat(original_tiles_parts, ignore_index=True)
    original_slides = pd.concat(original_slides_parts, ignore_index=True)

    num_workers = config.dataloader.num_workers
    batch_size = config.dataloader.batch_size
    persistent_workers = config.dataloader.persistent_workers and num_workers > 0

    all_slide_ids: list[str] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []

    for slide_dataset in dataset.generate_datasets():
        n_tiles = len(slide_dataset)

        try:
            slide_tiles_dataloader = DataLoader(
                slide_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            )

            slide_embeddings = torch.zeros(
                (n_tiles, tile_encoder.embed_dim), dtype=torch.float32
            )
            slide_x = torch.zeros((n_tiles,), dtype=torch.int32)
            slide_y = torch.zeros((n_tiles,), dtype=torch.int32)
            slide_ids: list[str] = [""] * n_tiles

            for i, (x, metadata) in enumerate(slide_tiles_dataloader):
                x = x.to(device)
                embeddings = cast("torch.Tensor", tile_encoder(x))

                start = i * batch_size
                end = start + embeddings.size(0)

                slide_embeddings[start:end] = embeddings.to("cpu")
                slide_x[start:end] = metadata["x"].to("cpu")
                slide_y[start:end] = metadata["y"].to("cpu")
                slide_ids[start:end] = list(metadata["slide_id"])

            all_slide_ids.extend(slide_ids)
            all_x.append(slide_x.numpy())
            all_y.append(slide_y.numpy())
            all_embeddings.append(slide_embeddings.numpy())

        except Exception:
            import traceback

            traceback.print_exc()
            sys.stdout.flush()

    embeddings_df = pd.DataFrame(
        {
            "slide_id": all_slide_ids,
            "x": np.concatenate(all_x).astype(np.int32),
            "y": np.concatenate(all_y).astype(np.int32),
            "embedding": list(np.concatenate(all_embeddings)),
        }
    )

    result_tiles = original_tiles.merge(
        embeddings_df, on=["slide_id", "x", "y"], how="left"
    )
    result_tiles.to_parquet(output_tiles_path, index=False, engine="pyarrow")
    original_slides.to_parquet(output_slides_path, index=False, engine="pyarrow")

    logger.log_artifacts(str(dest), artifact_path="embeddings")


if __name__ == "__main__":
    main()

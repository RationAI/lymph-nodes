r"""Inference pipeline: embeddings -> model -> tile probabilities -> heatmap TIFFs.

Usage:
    uv run -m postprocessing.run_inference \
        --embeddings-uri "mlflow-artifacts:/68/<run_id>/artifacts/embeddings" \
        --model-uri "mlflow-artifacts:/68/<run_id>/artifacts/checkpoints/epoch=1-step=74688"

    The model architecture config is pulled automatically from the same MLflow
    run that owns the checkpoint (from artifacts/configs/config.yaml, logged by
    autolog at training time).

    Optional flags:
        --slide-id KOS02_1234  process only this one slide (by parquet stem)
        --all-slides           process every parquet (default: first parquet only)
        --tile-extent 224      (pixels per tile side, default: 224)
        --mpp 0.5              (microns per pixel, default: 0.5)
        --heatmap-dest ./my_heatmaps
"""

import argparse
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


# Tile parameters - defaults matching configs/preprocessing/tiling.yaml
DEFAULT_TILE_EXTENT = 224
DEFAULT_MPP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on embeddings and produce heatmaps."
    )
    parser.add_argument(
        "--embeddings-uri",
        required=True,
        help="MLflow artifact URI pointing to the embeddings directory "
        "(contains one .parquet per slide).",
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow artifact URI of the checkpoint folder "
        "(mlflow-artifacts:/68/<run_id>/artifacts/checkpoints/<name>). "
        "The run_id is extracted automatically to fetch the training config.",
    )
    parser.add_argument(
        "--slide-id",
        default=None,
        help="Process only the parquet whose stem matches this slide ID. "
        "If omitted, only the first parquet found is processed.",
    )
    parser.add_argument(
        "--all-slides",
        action="store_true",
        help="Process every parquet in the embeddings directory "
        "(default: single-slide mode for quick testing).",
    )
    parser.add_argument(
        "--tile-extent",
        type=int,
        default=DEFAULT_TILE_EXTENT,
        help=f"Tile size in pixels (default: {DEFAULT_TILE_EXTENT}).",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=DEFAULT_MPP,
        help=f"Microns per pixel (default: {DEFAULT_MPP}).",
    )
    parser.add_argument(
        "--heatmap-dest",
        default="./my_heatmaps",
        help="Local directory for heatmap TIFFs (default: ./my_heatmaps).",
    )
    return parser.parse_args()


def _register_omegaconf_safe_globals() -> None:
    """Allow OmegaConf types embedded in Lightning checkpoints to be unpickled.

    PyTorch 2.6's weights_only=True default blocks them otherwise.
    """
    import omegaconf.base
    import omegaconf.nodes
    from omegaconf import DictConfig, ListConfig

    torch.serialization.add_safe_globals(
        [
            DictConfig,
            ListConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Metadata,
            omegaconf.nodes.ValueNode,
            omegaconf.nodes.BooleanNode,
            omegaconf.nodes.BytesNode,
            omegaconf.nodes.EnumNode,
            omegaconf.nodes.FloatNode,
            omegaconf.nodes.IntegerNode,
            omegaconf.nodes.StringNode,
            omegaconf.nodes.InterpolationResultNode,
        ]
    )


def main() -> None:
    args = parse_args()

    # -- Phase 1: PyTorch Inference ----------------------------------------
    print("Phase 1: Downloading embeddings & running inference...")

    embeddings_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=args.embeddings_uri,
    )

    parquet_files = sorted(Path(embeddings_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in downloaded artifact: {embeddings_dir}"
        )

    print(f"  Found {len(parquet_files)} slide embedding file(s).")

    # Filter to a single parquet for quick test runs
    if args.all_slides:
        pass  # use full list
    elif args.slide_id is not None:
        matched = [p for p in parquet_files if p.stem == args.slide_id]
        if not matched:
            available = ", ".join(p.stem for p in parquet_files)
            raise FileNotFoundError(
                f"No parquet found for slide_id '{args.slide_id}'. "
                f"Available: {available}"
            )
        parquet_files = matched
    else:
        parquet_files = parquet_files[:1]
        print(
            f"  Single-slide mode: processing '{parquet_files[0].stem}' only. "
            f"Pass --all-slides to process all slides."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _register_omegaconf_safe_globals()

    # autolog saves the full Hydra config to artifacts/configs/config.yaml in
    # the training run. Parse the model_run_id out of the model URI and fetch it.
    from lymph_nodes.meta_arch import MetaArch

    # mlflow-artifacts:/68/<model_run_id>/artifacts/...
    uri_parts = args.model_uri.split("/")
    model_run_id = uri_parts[2]  # mlflow-artifacts:/68/<run_id>/artifacts/...
    #                                              0  1    2        3
    # Use run_id + artifact_path form to avoid the double-artifacts URL bug
    # that occurs when building URI strings with an explicit artifacts/ segment.
    config_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=model_run_id,
            artifact_path="configs/config.yaml",
        )
    )
    model_cfg = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(model_cfg.model, _target_=MetaArch)

    ckpt_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=args.model_uri))
    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")

    ckpt = torch.load(ckpt_files[0], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    all_tiles = []

    for pq_path in parquet_files:
        slide_id = pq_path.stem
        embeddings_df = pd.read_parquet(pq_path)

        embeddings = np.stack(embeddings_df["embedding"].values)
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        slide_tiles = pd.DataFrame(
            {
                "slide_id": slide_id,
                "x": embeddings_df["x"].values,
                "y": embeddings_df["y"].values,
                "probability": probabilities,
            }
        )
        all_tiles.append(slide_tiles)
        print(f"    {slide_id}: {len(slide_tiles)} tiles")

    tiles_df = pd.concat(all_tiles, ignore_index=True)
    print(f"  Total: {len(tiles_df)} tiles across {len(parquet_files)} slides.")

    # -- Phase 2: Heatmap Generation ---------------------------------------
    print("Phase 2: Generating heatmaps...")

    from postprocessing.prediction_heatmap import prediction_heatmap

    slide_records = []
    for slide_id, group in tiles_df.groupby("slide_id"):
        slide_records.append(
            {
                "id": slide_id,
                "path": f"{slide_id}.mrxs",
                "extent_x": int(group["x"].max()) + args.tile_extent,
                "extent_y": int(group["y"].max()) + args.tile_extent,
                "tile_extent_x": args.tile_extent,
                "tile_extent_y": args.tile_extent,
                "mpp_x": args.mpp,
                "mpp_y": args.mpp,
            }
        )

    slides_df = pd.DataFrame(slide_records)

    mlflow.set_experiment("Lymph Nodes")

    with mlflow.start_run() as inference_run:
        prediction_heatmap(slides_df, tiles_df, dest=args.heatmap_dest)

    print(f"  MLflow run_id: {inference_run.info.run_id}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()

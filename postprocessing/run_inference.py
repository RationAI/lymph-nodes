r"""Inference pipeline: PyTorch inference -> Heatmap generation -> Pixel evaluation.

Usage:
    uv run -m postprocessing.run_inference \
        --embeddings-uri "mlflow-artifacts:/68/<run_id>/artifacts/embeddings" \
        --model-uri "mlflow-artifacts:/68/<run_id>/artifacts/checkpoints/epoch=1-step=74688"

    Optional flags:
        --tile-extent 224   (pixels per tile side, default: 224)
        --mpp 0.5           (microns per pixel, default: 0.5)
        --heatmap-dest ./my_heatmaps
        --skip-evaluation   (skip Phase 3)
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch


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
        help="MLflow artifact URI of the PyTorch checkpoint.",
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
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip Phase 3 (pixel evaluation).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Phase 1: PyTorch Inference ───────────────────────────────────────
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(args.model_uri, map_location=device)
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

    # ── Phase 2: Heatmap Generation ──────────────────────────────────────
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

    with mlflow.start_run() as run:
        prediction_heatmap(slides_df, tiles_df, dest=args.heatmap_dest)
        run_id = run.info.run_id

    print(f"  MLflow run_id: {run_id}")

    # ── Phase 3: Pixel Evaluation ────────────────────────────────────────
    if args.skip_evaluation:
        print("Phase 3: Skipped (--skip-evaluation).")
    else:
        print("Phase 3: Running pixel evaluation...")
        from postprocessing.evaluation import main as evaluate

        evaluate(seg_run_ids=[], cls_run_ids=[run_id])

    print("Pipeline complete.")


if __name__ == "__main__":
    main()

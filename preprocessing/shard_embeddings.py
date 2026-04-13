import os
import shutil
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


# Captured at import time, before @autolog changes the global tracking URI.
_SOURCE_TRACKING_URI = mlflow.get_tracking_uri()


_MLFLOW_SCHEMES = ("mlflow-artifacts:", "runs:/", "mlflow-run-artifacts:")


def resolve_path(uri_or_path: str) -> str:
    """Resolve an MLflow artifact URI or a plain disk path to a local file path."""
    if any(uri_or_path.startswith(scheme) for scheme in _MLFLOW_SCHEMES):
        return download_artifacts(
            artifact_uri=uri_or_path, tracking_uri=_SOURCE_TRACKING_URI
        )
    return uri_or_path


def shard_parquet(
    input_file: str,
    output_dir: str,
    rows_per_shard: int = 50_000,
    row_group_size: int = 5_000,
) -> None:
    """Stream-shard a large parquet file into smaller files without loading it fully into RAM."""
    os.makedirs(output_dir, exist_ok=True)

    parquet_file = pq.ParquetFile(input_file)
    print(f"Total rows in source: {parquet_file.metadata.num_rows}")

    shard_idx = 0
    current_shard_rows = 0
    writer = None

    for batch in parquet_file.iter_batches(batch_size=row_group_size):
        if writer is None:
            out_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.parquet")
            writer = pq.ParquetWriter(out_path, batch.schema)

        writer.write_batch(batch, row_group_size=row_group_size)
        current_shard_rows += batch.num_rows

        if current_shard_rows >= rows_per_shard:
            writer.close()
            writer = None
            print(f"Finished writing shard {shard_idx:05d}")
            shard_idx += 1
            current_shard_rows = 0

    if writer is not None:
        writer.close()
        print(f"Finished writing final shard {shard_idx:05d}")

    print("Sharding complete!")


@with_cli_args(["+preprocessing=shard_embeddings"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    embeddings_dir = resolve_path(config.embeddings_uri)

    tiles_path = os.path.join(embeddings_dir, "tiles.parquet")
    slides_path = os.path.join(embeddings_dir, "slides.parquet")
    print(f"tiles  → {tiles_path}")
    print(f"slides → {slides_path}")

    with TemporaryDirectory() as tmp_dir:
        print("Sharding tiles.parquet…")
        shard_parquet(tiles_path, tmp_dir, config.rows_per_shard, config.row_group_size)

        slides_dest = os.path.join(tmp_dir, "slides.parquet")
        print(f"Copying slides.parquet → {slides_dest}")
        shutil.copy2(slides_path, slides_dest)

        print("Uploading to MLflow…")
        logger.log_artifacts(local_dir=tmp_dir, artifact_path=config.artifact_path)

    print("Done!")


if __name__ == "__main__":
    main()

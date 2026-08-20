import hashlib
import json
import logging
from functools import cached_property
from typing import Any

import pyarrow.dataset as ds
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema


_logger = logging.getLogger(__name__)


class ParquetDataset(Dataset):
    """Lazy-loaded Parquet dataset (single file or sharded directory) with MLflow tracking."""

    def __init__(
        self,
        path: str,
        source: DatasetSource,
        target_col: str | None = None,
        name: str | None = None,
        digest: str | None = None,
    ) -> None:
        self._path = path
        self._target_col = target_col
        self._ds = ds.dataset(self._path, format="parquet")
        super().__init__(source=source, name=name, digest=digest)

    # ── MLflow Dataset interface ──────────────────────────────────────────────

    @property
    def data_type(self) -> str:
        return "parquet"

    @property
    def source(self) -> DatasetSource:
        return self._source

    @property
    def target_col(self) -> str | None:
        return self._target_col

    @property
    def dataset(self) -> ds.Dataset:
        return self._ds

    # ── digest ────────────────────────────────────────────────────────────────

    def _compute_digest(self) -> str:
        """Fast metadata-based digest — hashes schema + sorted file paths, never reads data."""
        hasher = hashlib.md5()
        hasher.update(str(self._ds.schema).encode())
        for f in sorted(self._ds.files):
            hasher.update(f.encode())
        return hasher.hexdigest()

    # ── profile ───────────────────────────────────────────────────────────────

    @property
    def profile(self) -> dict[str, Any]:
        """Row counts and structural metadata read from Parquet footers (no data blocks loaded)."""
        total_rows = 0
        for fragment in self._ds.get_fragments():
            if hasattr(fragment, "metadata") and fragment.metadata is not None:
                total_rows += fragment.metadata.num_rows
            else:
                total_rows += fragment.count_rows()
        return {
            "num_files": len(self._ds.files),
            "total_rows": total_rows,
            "num_columns": len(self._ds.schema.names),
            "backend_format": "parquet",
        }

    # ── schema ────────────────────────────────────────────────────────────────

    @cached_property
    def schema(self) -> Schema:
        try:
            empty_df = self._ds.head(0).to_pandas()
            # Drop list/array columns (e.g. embeddings) — MLflow schema cannot represent them
            scalar_cols = [
                name for name, field in zip(self._ds.schema.names, self._ds.schema)
                if not str(field.type).startswith(("list", "fixed_size_list", "large_list"))
            ]
            return _infer_schema(empty_df[scalar_cols])
        except Exception as exc:
            _logger.warning("Failed to infer schema for Parquet dataset: %s", exc)
            return Schema([])

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, str]:
        config = super().to_dict()
        config.update({
            "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()}),
            "profile": json.dumps(self.profile),
        })
        return config


# ── factory ───────────────────────────────────────────────────────────────────

def from_parquet(
    path: str,
    source: str | DatasetSource | None = None,
    target_col: str | None = None,
    name: str | None = None,
    digest: str | None = None,
) -> ParquetDataset:
    """Construct a ParquetDataset from a single file or a directory of shards.

    Example::

        dataset = from_parquet("/path/to/tiles/", target_col="tumor")
        mlflow.log_input(dataset, context="tiles")
    """
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.tracking.context import registry

    if source is not None:
        resolved_source = source if isinstance(source, DatasetSource) else resolve_dataset_source(source)
    else:
        resolved_source = CodeDatasetSource(tags=registry.resolve_tags())

    return ParquetDataset(path=path, source=resolved_source, target_col=target_col, name=name, digest=digest)

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
    def schema(self) -> Schema | None:
        try:
            import numpy as np
            import pyarrow as pa
            from mlflow.types.schema import TensorSpec

            pa_schema = self._ds.schema

            _NUMPY_DTYPES: dict[pa.DataType, np.dtype] = {
                pa.float16(): np.dtype("float16"),
                pa.float32(): np.dtype("float32"),
                pa.float64(): np.dtype("float64"),
                pa.int8():    np.dtype("int8"),
                pa.int16():   np.dtype("int16"),
                pa.int32():   np.dtype("int32"),
                pa.int64():   np.dtype("int64"),
                pa.uint8():   np.dtype("uint8"),
                pa.uint16():  np.dtype("uint16"),
                pa.uint32():  np.dtype("uint32"),
                pa.uint64():  np.dtype("uint64"),
                pa.bool_():   np.dtype("bool"),
            }

            def _is_scalar(t: pa.DataType) -> bool:
                return (
                    pa.types.is_integer(t) or pa.types.is_floating(t)
                    or pa.types.is_boolean(t) or pa.types.is_string(t)
                    or pa.types.is_large_string(t) or pa.types.is_binary(t)
                    or pa.types.is_date(t) or pa.types.is_timestamp(t)
                )

            def _tensor_spec(field: pa.Field) -> TensorSpec | None:
                """Try to build a TensorSpec for fixed-shape array/tensor columns."""
                t = field.type
                # Ray's ArrowTensorType and PyArrow's fixed_shape_tensor both expose
                # .shape; Ray also exposes .dtype, PyArrow uses .value_type.
                if pa.types.is_extension(t):
                    if hasattr(t, "shape"):
                        shape = tuple(t.shape)
                        np_dtype = (
                            np.dtype(t.dtype) if hasattr(t, "dtype")
                            else _NUMPY_DTYPES.get(t.value_type)  # type: ignore[attr-defined]
                        )
                        if np_dtype is not None:
                            return TensorSpec(np_dtype, shape=shape, name=field.name)
                    t = t.storage_type  # fall through to fixed_size_list unwrapping
                # Plain fixed_size_list (e.g. embedding stored as FSL<float32>[1536])
                dims: list[int] = []
                while pa.types.is_fixed_size_list(t):
                    dims.append(t.list_size)
                    t = t.value_type
                if dims:
                    np_dtype = _NUMPY_DTYPES.get(t)
                    if np_dtype is not None:
                        return TensorSpec(np_dtype, shape=tuple(dims), name=field.name)
                return None

            scalar_fields = [f for f in pa_schema if _is_scalar(f.type)]
            tensor_specs = [
                spec for f in pa_schema
                if not _is_scalar(f.type)
                if (spec := _tensor_spec(f)) is not None
            ]

            if not scalar_fields and not tensor_specs:
                return None

            specs: list = tensor_specs
            if scalar_fields:
                empty_table = pa.table({f.name: pa.array([], type=f.type) for f in scalar_fields})
                scalar_schema = _infer_schema(empty_table.to_pandas())
                specs = list(scalar_schema.inputs) + specs

            return Schema(specs)
        except Exception as exc:
            _logger.warning("Failed to infer schema for Parquet dataset: %s", exc)
            return None

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, str]:
        config = super().to_dict()
        if self.schema is not None:
            config["schema"] = json.dumps({"mlflow_colspec": self.schema.to_dict()})
        config["profile"] = json.dumps(self.profile)
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

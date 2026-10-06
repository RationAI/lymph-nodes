import logging
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.mlflow.parquet_dataset import from_parquet
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.utils import row_hash

from preprocessing.tiling_blocks import OverlayCoverage, TilingBlock


OmegaConf.register_new_resolver("scale", lambda x, factor: x * factor)

log = logging.getLogger(__name__)


def available_cpus() -> int:
    """Number of CPUs this process may run on.

    os.cpu_count() (and Ray's own detection) reports the whole node, while Kubernetes
    pins the pod to a subset of cores rather than setting a CFS quota.
    """
    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def drop_slides_without_mandatory_masks(
    slides_df: pd.DataFrame, tiling_blocks: list[TilingBlock], dataset_name: str
) -> pd.DataFrame:
    """Drop slides that some mandatory mask doesn't cover, reporting each one.

    OverlayCoverage would drop all their tiles anyway; doing it here makes that visible
    (a warning plus a ``missing_masks/<dataset>.json`` MLflow artifact) and skips tiling
    them at all. Raises if no slide is left, since an empty dataset makes Ray Data's
    streaming repartition wait forever instead of finishing.
    """
    missing: dict[str, list[str]] = {}
    for block in tiling_blocks:
        if isinstance(block, OverlayCoverage) and block.mandatory:
            paths = [p for p in slides_df["slide_path"] if not block.has_mask(p)]
            if paths:
                missing[block.name] = paths

    if not missing:
        return slides_df

    for name, paths in missing.items():
        log.warning(
            "%s: %d/%d slides have no mandatory '%s' mask and are skipped: %s",
            dataset_name,
            len(paths),
            len(slides_df),
            name,
            ", ".join(Path(p).name for p in paths),
        )
    mlflow.log_dict(missing, f"missing_masks/{dataset_name}.json")

    dropped = {p for paths in missing.values() for p in paths}
    kept = slides_df[~slides_df["slide_path"].isin(dropped)]
    if kept.empty:
        raise ValueError(
            f"{dataset_name}: no slide has all mandatory masks ({', '.join(missing)}), nothing to tile"
        )
    return kept


# ── tile generation ───────────────────────────────────────────────────────────

def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "path": row["path"],
        "slide_id": row["id"],
        "mpp_x": row["mpp_x"],
        "mpp_y": row["mpp_y"],
        "tile_extent_x": row["tile_extent_x"],
        "tile_extent_y": row["tile_extent_y"],
        "level": row["level"],
    }
    return [
        {"tile_x": int(x), "tile_y": int(y), **common}
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


def tile_dataset(
    slides_df: pd.DataFrame,
    tiling_blocks: list[TilingBlock],
    tile_extent: tuple[int, int],
    stride: tuple[int, int],
    mpp: int,
    rows_per_shard: int,
    dataset_name: str,
    project_root: Path,
) -> None:
    slide_paths = slides_df["slide_path"].tolist()
    meta_cols = [c for c in slides_df.columns if c != "slide_path"]

    slides_meta: dict[str, dict[str, Any]] = {
        row["slide_path"]: {k: row[k] for k in meta_cols}
        for _, row in slides_df.iterrows()
    }
    
    # --- Slide-level Ray Dataset ---
    slides = read_slides(slide_paths, mpp=mpp, tile_extent=tile_extent, stride=stride)
    slides = slides.map(row_hash)
    slides = slides.map(
        lambda r, meta=slides_meta: {**meta.get(r["path"], {}), **r},
    ).materialize()

    # --- Tile grid ---
    tiles = (
        slides
        .flat_map(tiling)
        .repartition(target_num_rows_per_block=512)
    )
    
    # --- Tiling blocks (sequential, config order) ---
    for block in tiling_blocks:
        tiles = block.apply(tiles)
    
    # --- Write sharded parquet + log to MLflow ---
    # Use a persistent staging directory so that a transient MLflow connection
    # failure after hours of computation doesn't lose the results.
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else "no_run"
    staging_dir = project_root / run_id / dataset_name
    tiles_dir = staging_dir / "tiles"
    slides_dir = staging_dir / "slides"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    tiles.repartition(target_num_rows_per_block=rows_per_shard).write_parquet(str(tiles_dir))
    slides.repartition(target_num_rows_per_block=rows_per_shard).write_parquet(str(slides_dir))

    mlflow.log_input(from_parquet(str(tiles_dir), name=dataset_name), context="tiles")
    mlflow.log_input(from_parquet(str(slides_dir), name=dataset_name), context="slides")


# ── entrypoint ────────────────────────────────────────────────────────────────

@with_cli_args(["+preprocessing=tiling"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # autolog redirects stdout to a file, so tqdm detects non-TTY and falls back to
    # printing a new line per update. Claiming isatty()=True makes it use \r-based
    # updates instead — the artifact then shows a compact in-place progress bar.
    # stdout only: tqdm writes to stderr by default, and wrapping stderr with a fake
    # TTY causes it to emit ANSI cursor-movement codes (\x1b[A) that appear as "[A"
    # garbage in the captured artifact file.
    # if not sys.stdout.isatty():
    #     _real_stdout = sys.stdout

    #     class _ForceTTY:
    #         def isatty(self) -> bool:
    #             return True
    #         def __getattr__(self, name: str) -> Any:
    #             return getattr(_real_stdout, name)

    #     sys.stdout = _ForceTTY()

    ray.init(num_cpus=available_cpus())

    data_context = ray.data.DataContext.get_current()
    for key, value in config.ray_data_context.items():
        if not hasattr(data_context, str(key)):
            raise ValueError(f"ray_data_context.{key} is not a ray.data.DataContext setting")
        setattr(data_context, str(key), value)

    tiling_blocks = [hydra.utils.instantiate(block_conf, _recursive_=False) for block_conf in config.tiling_blocks]
    project_root = Path(config.project_root)

    for i, dataset in enumerate(config.datasets):
        print(f"Tiling dataset: {dataset.name} ({i + 1}/{len(config.datasets)})")
        slides_df = hydra.utils.instantiate(dataset.slides).to_pandas()
        slides_df = drop_slides_without_mandatory_masks(slides_df, tiling_blocks, dataset.name)
        tile_dataset(
            slides_df=slides_df,
            tiling_blocks=tiling_blocks,
            tile_extent=config.tile_extent,
            stride=config.stride,
            mpp=config.mpp,
            rows_per_shard=config.rows_per_shard,
            dataset_name=dataset.name,
            project_root=project_root,
        )

    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else "no_run"
    logger.log_artifacts(str(project_root / run_id))

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

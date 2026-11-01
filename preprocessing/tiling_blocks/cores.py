from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from mlflow.artifacts import download_artifacts
from ray.data import Dataset

from preprocessing.tiling_blocks.tiling_block import TilingBlock


def _local(uri: str) -> Path:
    """A local directory or file for ``uri`` (downloaded unless it is a local path)."""
    return Path(uri) if Path(uri).exists() else Path(download_artifacts(uri))


@lru_cache(maxsize=16)
def label_mask(path: str) -> np.ndarray:
    """A core mask (uint16, a core's index per pixel), read once per worker."""
    with tifffile.TiffFile(path) as tiff:
        return tiff.pages[0].asarray()


class CoreSelection(TilingBlock):
    """Keeps the tiles of a split's TMA cores, and names the core of each.

    With cores split between folds (preprocessing/tma_split.py), a slide's row in a
    split's slides table lists its cores in that split (``cores``: indices in the core
    mask, ``;``-separated), and preprocessing.tiling passes the list on to its tiles.
    A tile belongs to the core under its centre in the core masks of
    preprocessing/tma_masks.py (``uri``: its ``tma_cores`` artifact, a uint16 mask per
    slide); tiles of other cores, and of none, are dropped. Adds ``name`` (the core's
    ``core_id`` from ``cores_uri``, that run's ``cores.csv``) and drops ``cores``.

    ``uri`` None, or tiles without ``cores`` (slides not split by core): every tile is
    kept, with a null ``name``.
    """

    def __init__(
        self, uri: str | None, cores_uri: str | None, name: str = "core_id"
    ) -> None:
        self._name = name
        self._masks: dict[str, str] = {}
        self._ids: dict[tuple[str, int], str] = {}
        self._mpp: dict[str, tuple[float, float]] = {}
        if uri is not None:
            self._masks = {p.stem: str(p) for p in _local(uri).rglob("*.tiff")}
        if cores_uri is not None:
            cores = pd.read_csv(_local(cores_uri))
            self._ids = {
                (row.slide, int(row.index)): row.core_id for row in cores.itertuples()
            }
            self._mpp = {
                row.slide: (float(row.label_mpp_x), float(row.label_mpp_y))
                for row in cores.drop_duplicates("slide").itertuples()
            }

    def apply(self, tiles: Dataset) -> Dataset:
        name, masks, ids, mpp = self._name, self._masks, self._ids, self._mpp

        def _select(batch: pd.DataFrame) -> pd.DataFrame:
            if "cores" not in batch.columns or not masks:
                batch[name] = None
                return batch.drop(columns=["cores"], errors="ignore")
            keep = np.ones(len(batch), bool)
            core_ids: list[str | None] = []
            for i, row in enumerate(batch.itertuples(index=False)):
                listed = row.cores
                slide = Path(row.path).stem
                if not isinstance(listed, str) or slide not in masks:
                    core_ids.append(None)  # not split by core
                    continue
                labels = label_mask(masks[slide])
                mx, my = mpp[slide]
                x = int((row.tile_x + row.tile_extent_x / 2) * row.mpp_x / mx)
                y = int((row.tile_y + row.tile_extent_y / 2) * row.mpp_y / my)
                inside = 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]
                index = int(labels[y, x]) if inside else 0
                wanted = {int(c) for c in listed.split(";") if c}
                if index == 0 or index not in wanted:
                    keep[i] = False
                    core_ids.append(None)
                else:
                    core_ids.append(ids.get((slide, index)))
            batch = batch.assign(**{name: core_ids})[keep]
            return batch.drop(columns=["cores"])

        return tiles.map_batches(_select, batch_format="pandas")

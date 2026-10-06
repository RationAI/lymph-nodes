from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from ray.data import Dataset
from shapely import Polygon


class OverlayCoverage:
    """Computes per-tile coverage from a pre-built mask directory (MLflow artifact URI).

    Coverage = sum of per-tile pixel fractions for mask labels in
    ``[min_label, max_label]`` (``max_label=None`` means unbounded above). Defaults
    to ``min_label=1, max_label=None`` — "everything except background label 0" —
    which is the right choice for a binary mask (e.g. ``JSONPolygonMask``'s 0/255
    output) but also lets a multi-level mask (severity buckets, several distinct
    ignore-reason codes, ...) define coverage over just the labels that actually
    matter, e.g. ``min_label=2, max_label=2`` to count only one specific class.
    Downloads the artifact directory eagerly so Ray workers never need MLflow credentials.

    Tiles whose slide has no corresponding mask file get a null coverage value instead
    of being dropped (unless ``mandatory=True``, in which case they're filtered out).
    ``uri=None`` means no mask source at all — every tile is treated as having no
    corresponding mask (same null-coverage behavior, just universal), without
    attempting a download. Useful for disabling a mask declaratively via config
    (e.g. ``ignore_mask_dir: null``) rather than removing the tiling block entirely.

    Pass ``tile_extent`` matching the global tiling config (use ``${tile_extent}``).
    """

    def __init__(
        self,
        name: str,
        uri: str | None,
        roi: Polygon | DictConfig,
        suffix: str = "tiff",
        mandatory: bool = False,
        min_label: int | None = None,
        max_label: int | None = None,
    ) -> None:
        from hydra.utils import instantiate

        self._name = name
        self._mandatory = mandatory
        self._roi = instantiate(roi) if isinstance(roi, DictConfig) else roi
        self._suffix = suffix
        self._min_label = min_label
        self._max_label = max_label
        if uri is None:
            self._overlay_paths = {}
        else:
            overlay_dir = download_artifacts(uri)
            self._overlay_paths = {p.stem: p for p in Path(overlay_dir).rglob(f"*.{suffix}")}

    def apply(self, tiles: Dataset) -> Dataset:
        from ratiopath.tiling import tile_overlay_overlap

        # The raw, undecorated implementation: eager on real pyarrow arrays, unlike
        # tile_overlay_overlap itself, which is a `ray.data.expressions` UDF that only
        # builds a lazy expression graph (see the `col(...)`-based usage it's designed for).
        # Calling it directly here keeps mask-path resolution and coverage computation
        # inside a single map_batches operator, so Ray Data never has to branch/replay
        # the dataset lineage to handle slides that have no corresponding mask file.
        compute_overlap = tile_overlay_overlap.__wrapped__

        overlay_paths = self._overlay_paths
        roi = self._roi
        name = self._name
        min_label = self._min_label
        max_label = self._max_label

        def in_range(label: str) -> bool:
            value = int(label)
            return (min_label is None or value >= min_label) and (
                max_label is None or value <= max_label
            )

        def add_coverage(batch: pa.Table) -> pa.Table:
            stems = [Path(p).stem for p in batch["path"].to_pylist()]
            mask_exists = pa.array([stem in overlay_paths for stem in stems], type=pa.bool_())

            with_mask = batch.filter(mask_exists)
            without_mask = batch.filter(pc.invert(mask_exists))

            if with_mask.num_rows:
                resolved_paths = pa.array(
                    [
                        str(overlay_paths[stem])
                        for stem, keep in zip(stems, mask_exists.to_pylist(), strict=True)
                        if keep
                    ],
                    type=pa.string(),
                )
                overlap = compute_overlap(
                    roi,
                    resolved_paths,
                    with_mask["tile_x"],
                    with_mask["tile_y"],
                    with_mask["mpp_x"],
                    with_mask["mpp_y"],
                )
                coverage = pa.array(
                    [
                        sum(
                            float(frac or 0.0)
                            for label, frac in (entry or {}).items()
                            if in_range(label)
                        )
                        for entry in overlap.to_pylist()
                    ],
                    type=pa.float64(),
                )
            else:
                coverage = pa.array([], type=pa.float64())

            with_mask = with_mask.append_column(name, coverage)
            without_mask = without_mask.append_column(name, pa.nulls(without_mask.num_rows, type=pa.float64()))

            return pa.concat_tables([with_mask, without_mask.select(with_mask.column_names)])

        tiles = tiles.map_batches(add_coverage, batch_format="pyarrow")

        if self._mandatory:
            tiles = tiles.filter(lambda r, c=name: r[c] is not None)

        return tiles

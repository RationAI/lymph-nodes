from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from mlflow.artifacts import download_artifacts
from ray.data import Dataset
from shapely import Polygon


class OverlayCoverage:
    """Computes per-tile coverage from a pre-built mask directory (MLflow artifact URI).

    Coverage = 1 - fraction_of_background_pixels (class "0" in the mask).
    Downloads the artifact directory eagerly so Ray workers never need MLflow credentials.

    Tiles whose slide has no corresponding mask file get a null coverage value instead
    of being dropped (unless ``mandatory=True``, in which case they're filtered out).

    Pass ``tile_extent`` matching the global tiling config (use ``${tile_extent}``).
    """

    def __init__(
        self,
        name: str,
        uri: str,
        roi: Polygon,
        suffix: str = "tiff",
        mandatory: bool = False,
    ) -> None:
        self._name = name
        self._mandatory = mandatory
        self._roi = roi
        self._suffix = suffix
        self._overlay_dir = download_artifacts(uri)
        self._overlay_stems = {p.stem for p in Path(self._overlay_dir).glob(f"*.{suffix}")}

    def apply(self, tiles: Dataset) -> Dataset:
        from ratiopath.tiling import tile_overlay_overlap

        # The raw, undecorated implementation: eager on real pyarrow arrays, unlike
        # tile_overlay_overlap itself, which is a `ray.data.expressions` UDF that only
        # builds a lazy expression graph (see the `col(...)`-based usage it's designed for).
        # Calling it directly here keeps mask-path resolution and coverage computation
        # inside a single map_batches operator, so Ray Data never has to branch/replay
        # the dataset lineage to handle slides that have no corresponding mask file.
        compute_overlap = tile_overlay_overlap.__wrapped__

        overlay_dir = self._overlay_dir
        overlay_stems = self._overlay_stems
        suffix = self._suffix
        roi = self._roi
        name = self._name

        def add_coverage(batch: pa.Table) -> pa.Table:
            stems = [Path(p).stem for p in batch["path"].to_pylist()]
            mask_exists = pa.array([stem in overlay_stems for stem in stems], type=pa.bool_())

            with_mask = batch.filter(mask_exists)
            without_mask = batch.filter(pc.invert(mask_exists))

            if with_mask.num_rows:
                overlay_paths = pa.array(
                    [
                        f"{overlay_dir}/{stem}.{suffix}"
                        for stem, keep in zip(stems, mask_exists.to_pylist(), strict=True)
                        if keep
                    ],
                    type=pa.string(),
                )
                overlap = compute_overlap(
                    roi,
                    overlay_paths,
                    with_mask["tile_x"],
                    with_mask["tile_y"],
                    with_mask["mpp_x"],
                    with_mask["mpp_y"],
                )
                coverage = pa.array(
                    [1.0 - float((entry or {}).get("0", 0.0) or 0.0) for entry in overlap.to_pylist()],
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

from pathlib import Path
from typing import Any

from mlflow.artifacts import download_artifacts
from ray.data import Dataset
from shapely import Polygon


class OverlayCoverage:
    """Computes per-tile coverage from a pre-built mask directory (MLflow artifact URI).

    Coverage = 1 - fraction_of_background_pixels (class "0" in the mask).
    Downloads the artifact directory eagerly so Ray workers never need MLflow credentials.

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
        self._files = {p.stem for p in Path(self._overlay_dir).glob(f"*.{suffix}")}

    def apply(self, tiles: Dataset) -> Dataset:
        from ratiopath.tiling import tile_overlay_overlap
        from ray.data.expressions import col as rcol

        path_col = f"__{self._name}_path"
        overlap_col = f"__{self._name}_overlap"

        def _add_path(row: dict[str, Any]) -> dict[str, Any]:
            overlay_path = Path(self._overlay_dir) / f"{Path(row["path"]).stem}.{self._suffix}"
            row[path_col] = str(overlay_path) if overlay_path.exists() else None
            return row

        def _extract(row: dict[str, Any]) -> dict[str, Any]:
            overlap = row.pop(overlap_col, {}) or {}
            row[self._name] = float(1.0 - (overlap.get("0", 0.0) or 0.0))
            return row

        def _compute_overlap(t: Dataset) -> Dataset:
            t = t.with_column(
                overlap_col,
                tile_overlay_overlap(self._roi, rcol(path_col), rcol("tile_x"), rcol("tile_y"), rcol("mpp_x"), rcol("mpp_y")),
            )
            return t.map(_extract)

        tiles = tiles.map(_add_path)

        if self._mandatory:
            tiles = tiles.filter(lambda r, c=path_col: r[c] is not None)
            tiles = _compute_overlap(tiles)
        else:
            has_mask = tiles.filter(lambda r, c=path_col: r[c] is not None)
            no_mask = tiles.filter(lambda r, c=path_col: r[c] is None)
            has_mask = _compute_overlap(has_mask)
            no_mask = no_mask.map(lambda r, c=self._name: {**r, c: None})
            tiles = has_mask.union(no_mask)

        return tiles.drop_columns([path_col, overlap_col])








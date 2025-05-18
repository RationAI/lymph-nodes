import tempfile
from pathlib import Path
import mlflow
import pandas as pd
from rationai.masks.mask_builders import ScalarMaskBuilder
from mlflow.artifacts import download_artifacts
import ray
from rationai.masks import (
    process_items,
)


def process_tiles(tiles, slide) -> None:
    tmp_dir = tempfile.TemporaryDirectory()

    mask_builder = ScalarMaskBuilder(
        save_dir=tmp_dir.name,
        filename=Path(slide.path).stem,
        extent_x=slide.extent_x,
        extent_y=slide.extent_y,
        mpp_x=slide.mpp_x,
        mpp_y=slide.mpp_y,
        stride=slide.stride_x,
        extent_tile=int(slide.tile_extent_x / 1.25),
    )

    mask_builder.update(
        (tiles["metastazis"] > 0) * 1,
        tiles["x"] + round(slide.tile_extent_y * 0.1),
        tiles["y"] + round(slide.tile_extent_y * 0.1),
    )

    pred_path = mask_builder.filename.with_suffix(".tiff")

    mlflow.log_artifact(
        str(pred_path),
        artifact_path=f"cls_gt_masks/{Path(slide.path).relative_to('/mnt/data/Projects/lymph_nodes/').parent}",
    )

    pred_path.unlink()
    tmp_dir.cleanup()


def main(uris: list[str]) -> None:
    mlflow.start_run(run_name="CLS eval GT mask")

    artifacts_paths = [download_artifacts(artifact_uri=uri) for uri in uris]

    slides_dfs: list[pd.DataFrame] = []
    tiles_dfs: list[pd.DataFrame] = []

    for path in artifacts_paths:
        if isinstance(path, str):
            path = Path(path)

        slides_dfs.append(pd.read_parquet(path / "slides.parquet"))
        tiles_dfs.append(pd.read_parquet(path / "tiles.parquet"))

    slides = pd.concat(slides_dfs)
    tiles = pd.concat(tiles_dfs)

    @ray.remote
    def process_slide(slide) -> None:
        slide_tiles = tiles[tiles["slide_id"] == slide.id]
        process_tiles(slide_tiles, slide)

    process_items(slides.itertuples(), process_item=process_slide)


if __name__ == "__main__":
    main(
        [
            "mlflow-artifacts:/68/5b263d24d75c4c138ffa33752a46e47c/artifacts/lymhps-2023/positive-test",
            "mlflow-artifacts:/68/5b263d24d75c4c138ffa33752a46e47c/artifacts/positive-lymph-nodes",
            "mlflow-artifacts:/68/5b263d24d75c4c138ffa33752a46e47c/artifacts/tmas/test",
        ]
    )

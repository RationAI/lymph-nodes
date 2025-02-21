from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pyvips
import ray
from openslide import OpenSlide
from rationai.masks import (
    process_items,
    tile_mask,
    write_big_tiff,
)

from preprocessing.utils import get_level_by_mpp, get_mpp, mpp_to_ppmm


RUN_ID = "61575574afb64f258c109f896c001054"


def create_slide_tiles(slide: Any, slide_tiles: pd.DataFrame) -> None:
    with OpenSlide(slide.path) as o_slide:
        level = get_level_by_mpp(o_slide, mpp=2)
        mpp = get_mpp(o_slide, level)
        xres, yres = mpp_to_ppmm(mpp)

    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide.tile_extent_x, slide.tile_extent_y),
        size=(slide.extent_x, slide.extent_y),
    )
    mask_path = Path("data/tile_masks") / f"{Path(slide.path).stem}.tiff"
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    write_big_tiff(
        pyvips.Image.new_from_array(mask), path=mask_path, xres=xres, yres=yres
    )


def create_tile_masks() -> None:
    # print("Downloading artifacts", flush=True)
    # mlflow.artifacts.download_artifacts(run_id=RUN_ID, dst_path="./data")

    slides: pd.DataFrame = pd.read_parquet("data/dataset/slides.parquet")
    tiles: pd.DataFrame = pd.read_parquet("data/dataset/tiles.parquet")

    tiles_ = ray.put(tiles)

    @ray.remote
    def process_slide(slide: Any) -> None:
        tiles = ray.get(tiles_)
        slide_tiles = tiles[tiles["slide_id"] == slide.id]
        create_slide_tiles(slide, slide_tiles)

    process_items(
        slides.iloc[[0, 100, 200, 300, 306, 310, 315]].itertuples(), process_slide, 1
    )

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="Tile - masks") as _:
        mlflow.log_artifacts("data/tile_masks")

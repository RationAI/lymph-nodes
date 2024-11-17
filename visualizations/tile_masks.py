from pathlib import Path
from typing import Any

import pandas as pd
import pyvips
import ray
from openslide import OpenSlide
from rationai.masks import process_items, tile_mask, write_big_tiff

from preprocessing.data import get_relative_dir_path
from preprocessing.utils import get_mpp, mpp_to_ppmm


def generate_mask(slide: Any, slide_tiles: pd.DataFrame, dest_dir: Path) -> None:
    with OpenSlide(slide.path) as slide:
        mpp = get_mpp(slide, slide.level)
        xres, yres = mpp_to_ppmm(mpp)

    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide.tile_extent_x, slide.tile_extent_y),
        size=(slide.extent_x, slide.extent_y),
    )
    mask_path = Path(dest_dir, f"{Path(slide.path).stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    write_big_tiff(
        pyvips.Image.new_from_array(mask), path=mask_path, xres=xres, yres=yres
    )


def tile_masks(slides: pd.DataFrame, tiles: pd.DataFrame) -> None:
    @ray.remote
    def process_slide(slide: Any) -> None:
        dest_dir = Path(
            "data/tile_mask", get_relative_dir_path(slide.path)
        )  # keep last level
        generate_mask(slide, tiles[tiles["slide_id"] == slide.id], dest_dir)

    process_items(slides.itertuples(), process_item=process_slide)

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from rationai.masks import process_items, write_big_tiff
from rationai.masks.slide_assembler import slide_assembler


def max_assembler(
    slide: Any,
    tiles: pd.DataFrame,
) -> pyvips.Image:
    def init(slide: Any) -> pyvips.Image:
        temp = np.memmap(
            f"{Path(slide.path).stem}_max.nmp",
            dtype=np.uint8,
            mode="w+",
            shape=(slide.extent_y, slide.extent_x),
        )
        return pyvips.Image.new_from_memory(
            temp.data, slide.extent_x, slide.extent_y, 1, format=pyvips.BandFormat.UCHAR
        )

    def aggregate(acc: pyvips.Image, slide: Any, tile: Any) -> pyvips.Image:
        return acc.draw_rect(
            255, tile.x, tile.y, slide.tile_extent_x, slide.tile_extent_y
        )

    def finalize(acc: pyvips.Image) -> pyvips.Image:
        return acc

    return slide_assembler(slide, tiles, init, aggregate, finalize)


def tile_visualization(slides: pd.DataFrame, tiles: pd.DataFrame, dest: str) -> None:
    @ray.remote
    def process_slide(slide: Any) -> None:
        slide_tiles = tiles[tiles["slide_id"] == slide.id]
        mask = max_assembler(slide, slide_tiles)

        mask_path = Path(dest, f"{Path(slide.path).stem}.tiff")
        mask_path.parent.mkdir(exist_ok=True, parents=True)

        write_big_tiff(
            mask,
            mask_path,
            mpp_x=slide.mpp_x,
            mpp_y=slide.mpp_y,
        )

    process_items(list(slides.itertuples()), process_item=process_slide)

    mlflow.log_artifacts(dest, "tile-visualizations")

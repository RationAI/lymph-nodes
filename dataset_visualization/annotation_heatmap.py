from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from numpy.typing import NDArray
from rationai.masks import process_items, write_big_tiff
from rationai.masks.slide_assembler import slide_assembler


def max_assembler(
    slide: Any,
    tiles: pd.DataFrame,
) -> pyvips.Image:
    def init(slide: Any) -> NDArray:
        return np.memmap(
            f"{Path(slide.path).stem}_annoatation-heat.nmp",
            dtype=np.int8,
            mode="w+",
            shape=(slide.extent_y, slide.extent_x),
        )

    def aggregate(acc: NDArray, slide: Any, tile: Any) -> NDArray:
        if tile.metastazis:
            acc[
                tile.y : tile.y + slide.tile_extent_y,
                tile.x : tile.x + slide.tile_extent_x,
            ] = 255
            acc.flush()
        return acc

    def finalize(acc: NDArray) -> pyvips.Image:
        return pyvips.Image.new_from_memory(
            acc.data, slide.extent_x, slide.extent_y, 1, format=pyvips.BandFormat.UCHAR
        )

    return slide_assembler(slide, tiles, init, aggregate, finalize)


def annotation_heatmap(slides: pd.DataFrame, tiles: pd.DataFrame, dest: str) -> None:
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

    mlflow.log_artifacts(dest, "annotation-heatmaps")

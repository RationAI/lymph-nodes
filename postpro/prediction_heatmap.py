from pathlib import Path
from typing import Any, TypeAlias

import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from numpy.typing import NDArray
from rationai.masks import process_items, write_big_tiff
from rationai.masks.slide_assembler import slide_assembler


Acc: TypeAlias = tuple[NDArray, NDArray]

FACTOR = 2**4


def norm(x: int) -> int:
    return round(x / FACTOR)


def heatmap_assembler_avg(slide: Any, tiles: pd.DataFrame) -> pyvips.Image:
    def init(slide: Any) -> Acc:
        heatmap = np.memmap(
            str(Path(slide.path).stem) + "_heatmap.nmp",
            dtype=np.float32,
            mode="w+",
            shape=(norm(slide.extent_y), norm(slide.extent_x)),
        )

        counts = np.memmap(
            str(Path(slide.path).stem) + "_count.nmp",
            dtype=np.uint8,
            mode="w+",
            shape=(norm(slide.extent_y), norm(slide.extent_x)),
        )

        return heatmap, counts

    def aggregate(acc: Acc, slide: Any, tile: Any) -> Acc:
        heatmap, counts = acc

        x, y = norm(tile.x), norm(tile.y)
        extent_x, extent_y = norm(slide.tile_extent_x), norm(slide.tile_extent_y)

        heatmap[y : y + extent_y, x : x + extent_x] += tile.probability
        counts[y : y + extent_y, x : x + extent_x] += 1

        heatmap.flush()
        counts.flush()

        return heatmap, counts

    def finalize(acc: Acc) -> pyvips.Image:
        heatmap, counts = acc

        counts = np.where(counts == 0, 1, counts)

        # t = np.asarray(np.where(counts != 0, heatmap / counts, 0) * 255, dtype=np.uint8)

        return pyvips.Image.new_from_array((heatmap / counts) * 255)

    return slide_assembler(slide, tiles, init, aggregate, finalize)


def prediction_heatmap(slides: pd.DataFrame, tiles: pd.DataFrame, dest: str) -> None:
    @ray.remote
    def process_slide(slide: Any) -> None:
        slide_tiles = tiles[tiles["slide_id"] == slide.id]

        if len(slide_tiles) == 0:
            return

        mask = heatmap_assembler_avg(slide, slide_tiles)

        mask_path = Path(dest, f"{Path(slide.path).stem}.tiff")
        mask_path.parent.mkdir(exist_ok=True, parents=True)

        write_big_tiff(
            mask,
            mask_path,
            mpp_x=slide.mpp_x,
            mpp_y=slide.mpp_y,
        )

    process_items(
        list(slides.itertuples()), process_item=process_slide, max_concurrent=10
    )

    mlflow.log_artifacts(dest, "heatmaps")

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from rationai.masks import process_items, write_big_tiff


DEST_DIR = "./data/heatmaps"


def heatmap(slide: Any, tiles: pd.DataFrame) -> None:
    heat = np.zeros((slide.extent_y, slide.extent_x)).astype(np.float32)
    counts = np.zeros((slide.extent_y, slide.extent_x)).astype(np.float32)

    for tile in tiles.itertuples():
        heat[tile.y : tile.y + slide.stride_y, tile.x : tile.x + slide.stride_x] += (
            tile.probability
        )
        counts[tile.y : tile.y + slide.stride_y, tile.x : tile.x + slide.stride_x] += 1

    counts[counts == 0] = 1
    heat = heat / counts
    heat = ((heat > 0.001) * 255).astype(np.uint8)

    mask_path = Path(DEST_DIR, f"{Path(slide.path).stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    write_big_tiff(pyvips.Image.new_from_array(heat), mask_path, 0, 0)


def create_heatmaps(slides: pd.DataFrame, tiles: pd.DataFrame) -> None:
    @ray.remote
    def process_slide(slide: Any) -> None:
        heatmap(slide, tiles[tiles["slide_id"] == slide.id])

    process_items(slides.itertuples(), process_item=process_slide, max_concurrent=2)

    mlflow.set_experiment(experiment_name="Lymph Nodes")
    with mlflow.start_run(run_name="DAB heatmaps") as _:
        mlflow.log_artifacts(DEST_DIR, "heatmaps")

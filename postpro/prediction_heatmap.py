from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pyvips
import ray
from rationai.masks import process_items, write_big_tiff
from rationai.masks.slide_assembler import heatmap_assembler_avg


def prediction_heatmap(
    slides: pd.DataFrame, predictions: pd.DataFrame, dest: str
) -> None:
    @ray.remote
    def process_slide(slide: Any) -> None:
        slide_predictions = predictions[predictions["slide_id"] == slide.id]
        mask = heatmap_assembler_avg(slide, slide_predictions, 4)

        mask_path = Path(dest, f"{Path(slide.path).stem}.tiff")
        mask_path.parent.mkdir(exist_ok=True, parents=True)

        write_big_tiff(
            mask,
            mask_path,
            mpp_x=slide.mpp_x,
            mpp_y=slide.mpp_y,
        )

    process_items(slides.itertuples(), process_item=process_slide)

    mlflow.log_artifacts(dest, "heatmaps")

import argparse
from typing import Literal

import mlflow

from preprocessing.calculate_mean_std import calculate_mean_std
from preprocessing.create_masks import create_inference_masks, create_training_masks
from preprocessing.tiling import inference_tiler, training_tiler
from visualizations.download_data import download_inference_data
from visualizations.heatmap import create_heatmaps


Triggres = Literal["create_masks", "tiling", "calculate_mean_std", "heatmap"]


def main(triggers: list[Triggres], params: dict) -> None:
    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000")
    # mlflow.set_tracking_uri("https://mlflow.rationai.cloud.trusted.e-infra.cz")

    if "create_masks" in triggers:
        create_training_masks()

    if "tiling" in triggers:
        training_tiler(run_with_masks_id=params["masks_run_id"])

    if "calculate_mean_std" in triggers:
        calculate_mean_std()

    if "heatmap" in triggers:
        slides, predictions = download_inference_data(
            params["inference_slides_uri"], params["predictions_uri"]
        )
        # slides = pd.read_parquet("./data/inference/slides.parquet")
        # predictions = pd.read_parquet("./data/inference/predictions.parquet")
        create_heatmaps(slides, predictions)


def inf_main(triggers: list[Triggres], params: dict) -> None:
    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000")
    # mlflow.set_tracking_uri("https://mlflow.rationai.cloud.trusted.e-infra.cz")

    if "create_masks" in triggers:
        create_inference_masks()

    if "tiling" in triggers:
        inference_tiler(run_with_masks_id=params["masks_run_id"])

    if "heatmap" in triggers:
        slides, predictions = download_inference_data(
            params["inference_slides_uri"], params["predictions_uri"]
        )
        # slides = pd.read_parquet("./data/inference/slides.parquet")
        # predictions = pd.read_parquet("./data/inference/predictions.parquet")
        create_heatmaps(slides, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some triggers.")
    parser.add_argument(
        "triggers", nargs="+", help="List of triggers with optional parameters"
    )
    parser.add_argument(
        "-masks_run_id", type=str, help="Run ID for tiling", default=None
    )
    parser.add_argument(
        "-inference_slides_uri", type=str, help="URI for inference slides", default=None
    )
    parser.add_argument(
        "-predictions_uri", type=str, help="URI for predictions", default=None
    )

    args = parser.parse_args()

    params = {}
    triggers = []
    for trigger in args.triggers:
        if trigger in Triggres.__args__:
            triggers.append(trigger)

    if "tiling" in triggers:
        params["masks_run_id"] = args.masks_run_id

    if "heatmap" in triggers:
        params["inference_slides_uri"] = args.inference_slides_uri
        params["predictions_uri"] = args.predictions_uri

    inf_main(triggers, params)

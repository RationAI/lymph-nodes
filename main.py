import argparse
from typing import Literal

import mlflow

from preprocessing import tiling
from preprocessing.calculate_mean_std import calculate_mean_std
from preprocessing.create_masks import create_masks


Triggres = Literal["create_masks", "tiling", "calculate_mean_std"]


def main(triggers: list[Triggres], params: dict) -> None:
    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000")

    if "create_masks" in triggers:
        create_masks()

    if "tiling" in triggers:
        tiling.tiler(run_with_masks_id=params["masks_run_id"])

    if "calculate_mean_std" in triggers:
        calculate_mean_std()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some triggers.")
    parser.add_argument(
        "triggers", nargs="+", help="List of triggers with optional parameters"
    )
    parser.add_argument(
        "-masks_run_id", type=str, help="Run ID for tiling", default=None
    )
    args = parser.parse_args()

    params = {}
    triggers = []
    for trigger in args.triggers:
        if trigger in Triggres.__args__:
            triggers.append(trigger)

    if "tiling" in triggers:
        params["masks_run_id"] = args.masks_run_id

    main(triggers, params)

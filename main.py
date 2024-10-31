import sys
from typing import Literal

import mlflow

from preprocessing import tiling
from preprocessing.calculate_mean_std import calculate_mean_std
from preprocessing.create_masks import create_masks
from preprocessing.create_tile_masks import create_tile_masks


Triggres = Literal["create_masks", "tiling", "create_slide_tiles", "calculate_mean_std"]


def main(triggers: list[Triggres]) -> None:
    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000")

    if "create_masks" in triggers:
        create_masks()

    if "tiling" in triggers:
        tiling.tiler()

    if "create_slide_tiles" in triggers:
        create_tile_masks()

    if "calculate_mean_std" in triggers:
        calculate_mean_std()


if __name__ == "__main__":
    triggers: list[Triggres] = [arg for arg in sys.argv[1:] if arg in Triggres.__args__]
    main(triggers)

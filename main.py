import sys
from typing import Literal

import mlflow

from preprocessing import tiling
from preprocessing.create_masks import create_masks


Triggres = Literal["create_masks", "tiling"]


def main(triggers: list[Triggres]) -> None:
    mlflow.set_tracking_uri("http://mlflow.rationai-mlflow:5000")

    if "create_masks" in triggers:
        create_masks()

    if "tiling" in triggers:
        tiling.tiler()


if __name__ == "__main__":
    triggers: list[Triggres] = [arg for arg in sys.argv[1:] if arg in Triggres.__args__]
    main(triggers)

from random import randint

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dataset_visualization.annotation_heatmap import annotation_heatmap
from dataset_visualization.tile_visualization import tile_visualization


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="./configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.metadata.mlflow_uri)
    mlflow.set_experiment(config.metadata.experiment_name)

    active_run = mlflow.start_run(run_name=config.metadata.run_name)

    print("Downloading artifacts")
    mlflow.artifacts.download_artifacts(
        artifact_uri=config.metadata.slides, dst_path="./data"
    )

    mlflow.artifacts.download_artifacts(
        artifact_uri=config.metadata.tiles, dst_path="./data"
    )

    print("Loading data")

    slides = pd.read_parquet(f"./data/slides.parquet")
    tiles = pd.read_parquet(f"./data/tiles.parquet")

    print("Creating heatmaps")
    # Prediction heatmaps
    annotation_heatmap(slides, tiles, config.metadata.heatmap_dest)

    # print("Create tile masks")
    # # Tile masks
    # tile_visualization(slides, tiles, config.metadata.tile_visualization_dest)

    mlflow.end_run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

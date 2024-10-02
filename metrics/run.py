# Copyright (c) The RationAI team

import os
from pathlib import Path

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from metrics import __version__


@hydra.main(
    version_base=None,
    config_path="../configs/preprocessing",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.mlflow_uri)
    experiment = mlflow.set_experiment(experiment_name=cfg.metadata.experiment_name)

    with mlflow.start_run(
        run_name=f"Metrics: {cfg.metadata.run_name}",
        experiment_id=experiment.experiment_id,
        tags={"version": __version__},
        description=cfg.metadata.description,
    ):
        mlflow.set_tag("mlflow.user", cfg.metadata.user)
        cfg_save_path = Path(os.getcwd()) / "config_resolved.yaml"
        OmegaConf.save(config=cfg, f=cfg_save_path, resolve=True)
        mlflow.log_artifact(str(cfg_save_path), artifact_path=".conf")
        mlflow.log_artifacts(".hydra", artifact_path=".conf")

        script = instantiate(cfg.scripts, _recursive_=True)
        script.log_params()
        script.calculate_metrics()

        mlflow.log_artifact(
            HydraConfig.get().job_logging.handlers.file.filename, artifact_path="logs"
        )


if __name__ == "__main__":
    main()

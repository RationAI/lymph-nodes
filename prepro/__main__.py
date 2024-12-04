from random import randint

import hydra
import mlflow
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="./configs", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    mlflow.set_tracking_uri(config.metadata.mlflow_uri)
    mlflow.set_experiment(config.metadata.experiment_name)
    with mlflow.start_run(run_name=config.metadata.run_name):
        for task in config.tasks:
            hydra.utils.instantiate(task, _recursive_=True)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

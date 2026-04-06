from random import randint

import hydra
import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import Logger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from omegaconf.dictconfig import DictConfig as OmegaDictConfig
from omegaconf.listconfig import ListConfig
from rationai.mlkit import Trainer, autolog

from lymph_nodes.data import DataModule


# PyTorch 2.6 changed torch.load default to weights_only=True.
# Checkpoints saved with older Lightning versions embed omegaconf objects in
# hyperparameters; register them as safe so deserialization succeeds.
torch.serialization.add_safe_globals([ListConfig, OmegaDictConfig])


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)
OmegaConf.register_new_resolver("key", lambda s: s.replace("-", "_"), use_cache=True)


@hydra.main(config_path="../configs", config_name="train_mil", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    seed_everything(config.seed, workers=True)

    kfold_splits = config.data.get("kfold_splits")
    k = config.data.get("k")

    if kfold_splits is not None and k is None:
        # Run all folds sequentially; each fold gets its own nested MLflow run.
        for fold in range(1, kfold_splits + 1):
            fold_config = OmegaConf.merge(config, {"data": {"k": fold}})
            run_name = (
                f"fold {fold}/{kfold_splits} — {config.foundation} {config.level}"
            )
            with mlflow.start_run(nested=True, run_name=run_name) as fold_run:
                fold_logger = MLFlowLogger(
                    run_id=fold_run.info.run_id,
                    experiment_name=config.metadata.experiment_name,
                )
                _run_fold(fold_config, fold_logger)
    else:
        _run_fold(config, logger)


def _run_fold(config: DictConfig, logger: Logger | None) -> None:
    data = hydra.utils.instantiate(
        config.data,
        _recursive_=False,
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    getattr(trainer, config.mode)(
        model, datamodule=data, ckpt_path=config.get("checkpoint")
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

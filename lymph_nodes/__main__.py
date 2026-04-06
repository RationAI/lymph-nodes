from random import randint

import hydra
import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import Logger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog

from lymph_nodes.data import DataModule


# PyTorch 2.6 changed torch.load default to weights_only=True.
# Checkpoints saved with older Lightning/omegaconf versions pickle arbitrary
# types (omegaconf internals, typing.Any, etc.) into save_hyperparameters().
# Those checkpoints come from our own trusted MLflow server, so loading with
# weights_only=False is safe.  Patch torch.load once here so every downstream
# call (including rationai-mlkit's trainer) picks it up without modification.
_original_torch_load = torch.load


def _torch_load_weights_only_false(f, *args, weights_only=True, **kwargs):
    return _original_torch_load(f, *args, weights_only=False, **kwargs)


torch.load = _torch_load_weights_only_false


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

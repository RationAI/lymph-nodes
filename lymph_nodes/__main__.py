# from functools import reduce
from random import randint
from typing import Literal

import hydra
import torch
from lightning import LightningDataModule, LightningModule, seed_everything
from lightning.pytorch.loggers import Logger
from lightning.pytorch.tuner.tuning import Tuner
from omegaconf import DictConfig, ListConfig, OmegaConf
from rationai.mlkit import Trainer, autolog


# from lymph_nodes import ClassificationModel, SegmentationModel
# from lymph_nodes.data import DataModule


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)

OmegaConf.register_new_resolver("model_name", lambda path: path.split(".")[-1])


# def load_dataset_config(task: str, kind: str) -> DictConfig:
#     """Dynamically loads the dataset config file."""
#     return reduce(
#         lambda conf, path: (DictConfig)(
#             OmegaConf.merge(
#                 conf,
#                 {
#                     path.split("/")[-1]: reduce(
#                         lambda conf, key: conf[key],
#                         path.split("/")[:-1],
#                         hydra.compose(config_name=path),
#                     )
#                 },
#             )
#         ),
#         map(
#             lambda x: f"data/datasets/{task}/{x}",
#             [f"{kind}/train", f"{kind}/val", "test", "predict"],
#         ),
#         OmegaConf.create(),
#     )


def find_batch_size(
    config: DictConfig,
    tuner: Tuner,
    model: LightningModule,
    method: Literal["fit", "validate", "test", "predict"],
    datamodule: LightningDataModule,
) -> int:
    """Finds the optimal batch size for the model."""
    if not config.dynamic_batch_size:
        return config.data.batch_size

    torch.backends.cudnn.enabled = False
    batch_size = tuner.scale_batch_size(
        model,
        method=method,
        mode="binsearch",
        datamodule=datamodule,
        steps_per_trial=1,
        max_trials=2,
    )
    torch.backends.cudnn.enabled = True

    print(f"Batch size for {method}: {batch_size}")

    return batch_size


@hydra.main(config_path="../configs", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = hydra.utils.instantiate(config.data, _recursive_=False)
    model = hydra.utils.instantiate(config.model)

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    tuner = hydra.utils.instantiate(config.tuner, _target_=Tuner, trainer=trainer)

    if isinstance(config.mode, ListConfig):
        for mode in config.mode:
            data.batch_size = find_batch_size(
                config,
                tuner,
                model,
                method=mode,
                datamodule=data,
            )

            getattr(trainer, mode)(
                model, datamodule=data, ckpt_path=config.checkpoint.get(mode, None)
            )
    else:
        data.batch_size = find_batch_size(
            config,
            tuner,
            model,
            method=config.mode,
            datamodule=data,
        )

        getattr(trainer, config.mode)(
            model, datamodule=data, ckpt_path=config.checkpoint
        )

    if "test" in config.mode:
        import subprocess

        # Replace the run_id in the config file it is not possible to replace all occuances with hydra
        command = (
            f"sed -i 's|\\${{run_id}}|{logger.run_id}|g' configs/report/default.yaml"
        )
        subprocess.run(command, shell=True, check=True)

        result = subprocess.run(
            [
                "python",
                "-m",
                "report",
                "--config-path=../../../../../configs/report",
                f"run_id={logger.run_id}",
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout, result.stderr)

    # data = hydra.utils.instantiate(
    #     OmegaConf.merge(
    #         OmegaConf.create(config.data),
    #         load_dataset_config(config.task, config.kind),
    #     ),
    #     _recursive_=False,  # to avoid instantiating all the datasets
    #     _target_=DataModule,
    # )
    # model = hydra.utils.instantiate(
    #     config.model,
    #     _target_=SegmentationModel
    #     if config.task == "segmentation"
    #     else ClassificationModel,
    # )

    # trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    # getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

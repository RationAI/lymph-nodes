# from functools import reduce
from random import randint

import hydra
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, ListConfig
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


@hydra.main(config_path="../configs", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = hydra.utils.instantiate(config.data, _recursive_=False)
    model = hydra.utils.instantiate(config.model)

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)

    if isinstance(config.mode, ListConfig):
        for mode in config.mode:
            getattr(trainer, mode)(
                model, datamodule=data, ckpt_path=config.checkpoint.get(mode, None)
            )
    else:
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

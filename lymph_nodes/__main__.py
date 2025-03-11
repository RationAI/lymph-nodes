from functools import reduce
from pathlib import Path
from random import randint

import hydra
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog

from lymph_nodes import ClassificationModel, SegmentationModel
from lymph_nodes.data import DataModule


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


def load_dataset_config(task: str, kind: str) -> DictConfig:
    """Dynamically loads the dataset config file."""
    return reduce(
        lambda conf, path: (DictConfig)(
            OmegaConf.merge(
                conf,
                {
                    Path(path).stem: OmegaConf.load(
                        f"./config/data/datasets/{task}/{path}"
                    )
                },
            )
        ),
        [f"{kind}/train.yaml", f"{kind}/val.yaml", "/test.yaml", "predict.yaml"],
        OmegaConf.create(),
    )


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    seed_everything(config.seed, workers=True)

    torch.set_float32_matmul_precision("medium")

    data = hydra.utils.instantiate(
        OmegaConf.merge(
            config.data,
            load_dataset_config(config.task, config.kind),
        ),
        _recursive_=False,  # to avoid instantiating all the datasets
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(
        config.model,
        _target_=SegmentationModel
        if config.task == "segmentation"
        else ClassificationModel,
    )

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

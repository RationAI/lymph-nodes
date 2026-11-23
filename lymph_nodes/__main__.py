import hydra
import torch
from lightning import seed_everything
from omegaconf import DictConfig
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from lymph_nodes import gpu_plan
from lymph_nodes.data import DataModule
from lymph_nodes.resolvers import register_resolvers


# Before @hydra.main composes the config, i.e. at import time, not inside main().
register_resolvers()


@hydra.main(config_path="../configs", config_name="lymph_nodes", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    gpu_plan.limit_memory()  # a share of the GPU, when gpu_plan.share started this run
    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(
        config.data,
        _recursive_=False,  # to avoid instantiating all the datasets
        _target_=DataModule,
    )
    # The model config names its class (MetaArch, SegmentationMetaArch, ...).
    model = hydra.utils.instantiate(config.model)

    extra = {}
    if config.mode == "fit" and config.gpu_plan.enabled and torch.cuda.is_available():
        plan = gpu_plan.apply(config, model, data)  # the batch that fits this GPU
        logger.log_hyperparams(plan.params())
        if plan.accumulate > 1:
            extra["accumulate_grad_batches"] = plan.accumulate
        seed_everything(config.seed, workers=True)  # the probe drew random batches

    trainer = hydra.utils.instantiate(
        config.trainer, _target_=Trainer, logger=logger, **extra
    )
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

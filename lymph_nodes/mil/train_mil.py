import logging
import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


# Ensure local modules are found
sys.path.append(os.getcwd())

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../configs/mil", config_name="train_mnist"
)
def main(cfg: DictConfig):
    log.info(f"Initializing MIL Experiment: {cfg.experiment_name}")

    # 1. Instantiate Data and Model using the _target_ defined in yaml
    dataset = instantiate(cfg.data)
    model = instantiate(cfg.model)

    # 2. Initialize Trainer
    # We import inside main to avoid circular dependencies if any
    from lymph_nodes.mil.trainer import MILTrainer

    trainer = MILTrainer(model, dataset, cfg)

    # 3. Run
    trainer.run()


if __name__ == "__main__":
    main()

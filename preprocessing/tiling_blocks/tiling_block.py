from __future__ import annotations

from abc import ABC, abstractmethod

from omegaconf import DictConfig
from ray.data import Dataset


class TilingBlock(ABC):
    @abstractmethod
    def apply(self, tiles: Dataset) -> Dataset: ...


def instantiate_tiling_block(cfg: DictConfig) -> TilingBlock:
    import hydra

    return hydra.utils.instantiate(cfg, _recursive_=True)

from __future__ import annotations

from abc import ABC, abstractmethod

from ray.data import Dataset


class TilingBlock(ABC):
    @abstractmethod
    def apply(self, tiles: Dataset) -> Dataset: ...


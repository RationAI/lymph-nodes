from typing import Any

from ray.data import Dataset

from preprocessing.tiling_blocks.tiling_block import TilingBlock


class Operator(TilingBlock):
    """Apply a named ``ray.data.Dataset`` method as a tiling-pipeline block."""

    def __init__(self, operator: str, *args: Any, **kwargs: Any) -> None:
        self._f_name = operator
        self._args = args
        self._kwargs = kwargs

    def apply(self, dataset: Dataset) -> Dataset:
        return getattr(dataset, self._f_name)(*self._args, **self._kwargs)
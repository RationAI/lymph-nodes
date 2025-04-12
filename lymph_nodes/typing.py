from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TypeAlias, TypedDict

from torch import Tensor


class Metadata(TypedDict):
    slide_id: bytes
    slide: str
    x: int
    y: int


class Prediction(TypedDict):
    slide_id: bytes
    x: int
    y: int
    probability: float


PredictSample: TypeAlias = tuple[Tensor, Metadata]
ClsSample: TypeAlias = tuple[Tensor, Tensor, Metadata]
SegSample: TypeAlias = tuple[Tensor, Tensor, Tensor, Metadata]

Input: TypeAlias = ClsSample | SegSample
PredictInput: TypeAlias = PredictSample


@dataclass
class Outputs(Iterable["Outputs"]):
    def __init__(
        self, labels: Tensor | None = None, masks: Tensor | None = None
    ) -> None:
        self.labels = labels
        self.masks = masks

    def __iter__(self) -> Iterator["Outputs"]:
        for i in range((self.labels or self.masks).shape[0]):
            label = self.labels[i] if self.labels else None
            mask = self.masks[i] if self.masks else None
            yield Outputs(labels=label, masks=mask)


Targets: TypeAlias = Outputs

Predictions: TypeAlias = list[Prediction]

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
Sample: TypeAlias = tuple[Tensor, Tensor, Metadata]

Input: TypeAlias = Sample
PredictInput: TypeAlias = PredictSample


Outputs: TypeAlias = Tensor

Predictions: TypeAlias = list[Prediction]

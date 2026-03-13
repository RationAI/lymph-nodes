from typing import TypedDict

from torch import Tensor


class TileMetadata(TypedDict):
    slide_id: str
    x: int
    y: int


type TilesPredictSample = tuple[Tensor, TileMetadata]
type TilesSample = tuple[Tensor, TileMetadata, Tensor]

type Input = tuple[Tensor, Tensor]  # TODO: define your model input type
type Outputs = Tensor  # TODO: define your model output type

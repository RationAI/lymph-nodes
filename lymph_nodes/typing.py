from typing import TypedDict

from torch import Tensor


class TileMetadata(TypedDict):
    slide_id: str
    tile_x: int
    tile_y: int


type TilesPredictSample = tuple[Tensor, TileMetadata]

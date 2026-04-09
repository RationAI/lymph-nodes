from pathlib import Path
from typing import TypedDict

import pandas as pd
from torch import Tensor


class Metadata(TypedDict):
    slide_id: str
    slide_name: str
    slide_path: Path


class MetadataTileEmbeddings(Metadata):
    slide_id: str
    slide_name: str
    slide_path: Path
    tiles: pd.DataFrame
    x: Tensor
    y: Tensor


type TileEmbeddingsSample = tuple[Tensor, Tensor, MetadataTileEmbeddings]
type TileEmbeddingsInput = tuple[Tensor, Tensor, list[MetadataTileEmbeddings]]

type Output = Tensor

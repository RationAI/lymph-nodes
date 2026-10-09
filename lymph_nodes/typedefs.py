from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from torch import Tensor


type EmbeddingInput = NDArray[np.float32]

# The cancer target: 0/1, or a soft target in between (TileLabels.soft_target).
type EmbeddingLabel = float

type Sample[I, O] = tuple[I, O, dict[str, Any]]

type EmbeddingSample = Sample[EmbeddingInput, EmbeddingLabel]

type Input = Tensor

type Outputs = Tensor

# What a DataLoader/MetaArch step actually receives: EmbeddingSample describes one
# uncollated sample (label as a plain float), but after collation the label becomes a
# Tensor too — a distinct type from EmbeddingSample, not just its "batched" form.
type Batch = tuple[Input, Tensor, dict[str, Any]]

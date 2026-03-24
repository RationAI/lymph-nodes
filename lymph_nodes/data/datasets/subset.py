from collections.abc import Sequence

from torch.utils.data import Subset

from lymph_nodes.data.datasets.tile_embeddings import TileEmbeddings
from lymph_nodes.typing import TileEmbeddingsSample


class TileEmbeddingsSubset(Subset[TileEmbeddingsSample]):
    def __init__(
        self,
        dataset: TileEmbeddings,
        indices: Sequence[int],
    ) -> None:
        super().__init__(dataset, indices)
        self.slides = [dataset.slides[i] for i in indices]
        self.labels = [dataset.labels[i] for i in indices]
        self.groups = [dataset.groups[i] for i in indices]


def create_subset(
    dataset: TileEmbeddings, indices: Sequence[int]
) -> TileEmbeddingsSubset:
    return TileEmbeddingsSubset(dataset, indices)

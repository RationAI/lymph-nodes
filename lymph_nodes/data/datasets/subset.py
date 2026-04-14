from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset, Subset

from lymph_nodes.data.datasets.tile_embeddings import TileEmbeddings
from lymph_nodes.typing import TileEmbeddingsSample


class TileEmbeddingsSubset(Subset[TileEmbeddingsSample]):
    def __init__(self, dataset: TileEmbeddings, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)
        self.slides = [dataset.slides[i] for i in indices]
        self.labels = [dataset.labels[i] for i in indices]
        self.groups = [dataset.groups[i] for i in indices]


class DatasetSubset(Subset[Any]):
    """Generic subset that preserves .slides, .labels, .groups attributes."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)
        self.labels = [dataset.labels[i] for i in indices]  # type: ignore[attr-defined]
        self.groups = [dataset.groups[i] for i in indices]  # type: ignore[attr-defined]

        if hasattr(dataset, "_tile_slide_names"):
            slide_lookup = {s["name"]: s for s in dataset.slides}  # type: ignore[attr-defined]
            seen: set[str] = set()
            self.slides: list[dict] = []
            for i in indices:
                slide_name = dataset._tile_slide_names[i]  # type: ignore[attr-defined]
                if slide_name not in seen:
                    seen.add(slide_name)
                    self.slides.append(slide_lookup[slide_name])
        else:
            self.slides = [dataset.slides[i] for i in indices]  # type: ignore[attr-defined]


def create_subset(
    dataset: Dataset, indices: Sequence[int]
) -> TileEmbeddingsSubset | DatasetSubset:
    if isinstance(dataset, TileEmbeddings):
        return TileEmbeddingsSubset(dataset, indices)
    return DatasetSubset(dataset, indices)

from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset, Subset


class DatasetSubset(Subset[Any]):
    """Generic subset that preserves .slides, .labels, .groups attributes."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)
        self.labels = [dataset.labels[i] for i in indices]  # type: ignore[attr-defined]
        self.groups = [dataset.groups[i] for i in indices]  # type: ignore[attr-defined]

        # For tile-level datasets (e.g. TilePatchDataset), dataset.slides contains
        # one entry per unique slide while indices are tile indices — direct indexing
        # would be out of range.  Rebuild the unique-slide list from tile metadata.
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


def create_subset(dataset: Dataset, indices: Sequence[int]) -> DatasetSubset:
    return DatasetSubset(dataset, indices)

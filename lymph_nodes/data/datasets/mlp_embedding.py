from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset


if TYPE_CHECKING:
    from collections.abc import Iterable


def _group_from_stem(stem: str) -> str:
    # 1. Standard SNB IHC/LN pattern (e.g., SNB_IHC_CASE_5_SLIDE_1-0, SNB_LN_CASE_42_SLIDE_1-1)
    snb_match = re.match(r"^SNB_(?:IHC|LN)_CASE_(\d+)_", stem)
    if snb_match:
        return f"snb_case_{snb_match.group(1)}"

    # 2. FNBrno pattern (e.g., FNB1206-23-4 or FNB-P1480-25-5)
    fnb_match = re.match(r"^FNB-?P?(\d+)-(\d+)-", stem)
    if fnb_match:
        return f"fnb_{fnb_match.group(1)}_{fnb_match.group(2)}"

    # 3. SNB TEST/Annotated pattern (e.g., SNB_IHC_TEST_CASE-2024_1011-15)
    test_match = re.search(r"TEST_CASE-(\d+)_(\d+-\d+)", stem)
    if test_match:
        return f"test_{test_match.group(1)}_{test_match.group(2)}"

    # 4. Default to using the full stem as the group if no pattern matches
    return stem


def _label_from_filename(stem: str) -> int:
    """Helper to extract slide-level labels from slide IDs."""
    if stem.endswith("-1"):
        return 1
    if stem.endswith("-0"):
        return 0
    return 1


class SlideEmbeddingDataset(Dataset):
    """The leaf dataset representing a single slide."""

    def __init__(
        self, tiles: HFDataset, name: str, label: int, metastazis_threshold: float = 0.0
    ) -> None:
        self._tiles = tiles.with_format("numpy")
        self.name = name
        self.label = label
        self._metastazis_threshold = metastazis_threshold

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self._tiles[idx]

        embedding = torch.from_numpy(row["embedding"].copy())
        label = torch.tensor(
            float(row["metastazis"] >= self._metastazis_threshold), dtype=torch.float32
        )

        metadata = {"x": row["x"], "y": row["y"], "slide_id": self.name}

        return embedding, label, metadata


class MLPEmbeddingDataset(MetaTiledSlides):
    def __init__(
        self,
        paths: list[str],
        uris: list[str] | None = None,
        metastazis_threshold: float = 0.0,
        shuffle_labels: bool = False,
    ) -> None:
        self._metastazis_threshold = metastazis_threshold
        self._shuffle_labels = shuffle_labels
        super().__init__(paths=paths, uris=uris)
        if shuffle_labels:
            permuted = np.random.permutation(self.tiles["metastazis"])
            self.tiles = self.tiles.remove_columns("metastazis").add_column(
                "metastazis", permuted.tolist()
            )
            self.datasets = list(
                self.generate_datasets()
            )  # ← re-create with shuffled tiles

    def generate_datasets(self) -> Iterable[Dataset]:
        for slide in self.slides:
            slide_id = slide["id"]
            slide_tiles_view = self.filter_tiles_by_slide(slide_id)

            if len(slide_tiles_view) > 0:
                yield SlideEmbeddingDataset(
                    tiles=slide_tiles_view,
                    name=slide_id,
                    label=_label_from_filename(slide_id),
                    metastazis_threshold=self._metastazis_threshold,
                )

    @property
    def labels(self) -> np.ndarray:
        """Required for WeightedRandomSampler."""
        return (
            np.array(self.tiles["metastazis"]) >= self._metastazis_threshold
        ).astype(np.int8)

    @property
    def groups(self) -> np.ndarray:
        """Required for StratifiedGroupKFold."""
        return np.array([_group_from_stem(sid) for sid in self.tiles["slide_id"]])

    @property
    def slides(self) -> Any:
        if hasattr(self, "datasets") and self.datasets:
            return [{"name": ds.name, "label": ds.label} for ds in self.datasets]

        return self.__dict__.get("slides")

    @slides.setter
    def slides(self, value: Any) -> None:
        """Allows the parent class to set self.slides during init."""
        self.__dict__["slides"] = value

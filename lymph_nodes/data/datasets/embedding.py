from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from torch.utils.data import Dataset

from lymph_nodes.data.datasets.meta_dataset import MetaDataset
from lymph_nodes.typedefs import EmbeddingSample


if TYPE_CHECKING:
    from collections.abc import Iterable

    from datasets import Dataset as HFDataset


# Partition ids used both for the per-tile label (0/1/2 collapsed to healthy/cancer in
# __getitem__) and for lymph_nodes.data.samplers.stratified_sampler's rebalancing.
# Keep in sync with that module's PARTITION_NAMES.
HEALTHY, HEALTHY_BROWNISH, CANCER = 0, 1, 2


def _is_cancer(annotation_coverage: Any, cytokeratin_coverage: Any) -> Any:
    """True if either overlay is present.

    Works on scalars or numpy arrays alike (hence ``Any`` rather than a precise
    scalar/array union, which would need an ``@overload`` pair for one line of logic)
    — NaN > 0 is False, so missing coverage (no mask at all for that slide) needs no
    separate null handling.
    """
    return (annotation_coverage > 0) | (cytokeratin_coverage > 0)


class TileEmbeddingClassificationDataset(Dataset[EmbeddingSample]):
    """One slide's tiles, with a binary cancer label derived from overlay coverage.

    There is no stored "label" column — it's computed per tile from coverage columns
    written by the preprocessing pipeline's OverlayCoverage blocks: a tile counts as
    cancer if the pathologist's cancer annotation overlaps it (``annotation_coverage``)
    OR the cytokeratin IHC epithelium overlay overlaps it (``cytokeratin_coverage``) —
    cytokeratin stains epithelial cells, which should not be present in healthy lymph
    node tissue, so cytokeratin positivity is treated as (metastatic) epithelium
    regardless of whether the region was also annotated.
    """

    def __init__(
        self,
        name: str,
        tiles: HFDataset,
    ):
         super().__init__()
         self.name = name
         self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> EmbeddingSample:
        tile = self.tiles[idx]
        has_cancer = bool(_is_cancer(tile["annotation_coverage"], tile["cytokeratin_coverage"]))
        return tile["embedding"], has_cancer, {"name": self.name}

    def partition_labels(self) -> np.ndarray:
        """Vectorized per-tile group id: HEALTHY, HEALTHY_BROWNISH, or CANCER.

        "Healthy brownish" is DAB/brown-stain coverage with no cancer overlay — tissue
        that resembles metastasis on staining alone but wasn't annotated or
        cytokeratin-positive. It's the hardest class to tell apart from true cancer,
        so it's split out as its own partition for
        lymph_nodes.data.samplers.stratified_sampler.StratifiedEpochSampler to
        oversample relative to its natural (rare) frequency, as a static stand-in for
        online hard-negative mining.
        """
        annotation = np.asarray(self.tiles["annotation_coverage"])
        cytokeratin = np.asarray(self.tiles["cytokeratin_coverage"])
        brownish = np.asarray(self.tiles["brownish_coverage"])

        cancer = _is_cancer(annotation, cytokeratin)
        healthy_brownish = ~cancer & (brownish > 0.2)

        labels = np.full(len(self.tiles), HEALTHY, dtype=np.int64)
        labels[healthy_brownish] = HEALTHY_BROWNISH
        labels[cancer] = CANCER
        return labels


class EmbeddingClassificationDataset(MetaDataset[EmbeddingSample]):
    def generate_datasets(self) -> Iterable[TileEmbeddingClassificationDataset]:
        return (TileEmbeddingClassificationDataset(
            name = slide["slide_name"],
            tiles = self.filter_tiles_by_slide(slide["id"])
        ) for slide in self.slides)
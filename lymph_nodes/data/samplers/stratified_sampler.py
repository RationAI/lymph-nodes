from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mlflow
import numpy as np
from torch.utils.data import Sampler

from lymph_nodes.data.datasets.embedding import CANCER, HEALTHY, HEALTHY_BROWNISH


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torch.utils.data import ConcatDataset

    from lymph_nodes.data.datasets.embedding import TileEmbeddingClassificationDataset


PARTITION_NAMES: dict[str, int] = {
    "healthy": HEALTHY,
    "healthy_brownish": HEALTHY_BROWNISH,
    "cancer": CANCER,
}


class StratifiedEpochSampler(Sampler[int]):
    """Draws a fresh random subset per partition each epoch, in a caller-chosen ratio.

    Partitions are healthy / healthy_brownish / cancer, drawn fresh instead of a full
    pass over every tile. Built for severe class imbalance (e.g. ~20:1 healthy:cancer):
    grinding through every negative tile each epoch wastes compute on the easy majority
    and still under-represents cancer tiles in every batch. Resampling a fresh subset
    each epoch — rather than a fixed one — means every negative tile is still eventually
    seen across training, just not all of them in any one epoch. ``healthy_brownish``
    (DAB-positive
    but not annotated or cytokeratin-positive — the tissue most visually confusable
    with real cancer) is split out from plain ``healthy`` so it can be deliberately
    oversampled relative to its natural frequency, as a static, no-feedback-loop
    stand-in for online hard-negative mining.

    Expects ``dataset.datasets`` (as built by ``torch.utils.data.ConcatDataset``,
    which ``MetaTiledSlides`` subclasses) to be
    ``TileEmbeddingClassificationDataset``-like: each must expose
    ``partition_labels()`` returning one ``HEALTHY`` / ``HEALTHY_BROWNISH`` / ``CANCER``
    code per tile, in the same order as that sub-dataset's own indexing — matching how
    ``ConcatDataset`` concatenates them into the global flat tile index.

    ``ratio`` is keyed by partition name (not forced equal — tune it to your actual
    imbalance), e.g. ``{"healthy": 2, "healthy_brownish": 2, "cancer": 1}``.

    ``epoch_length``, if not given, is auto-derived rather than defaulting to
    ``len(dataset)`` — the total tile count is dominated by the majority class and
    doesn't reflect what the rare partitions can actually support. Instead it's set to
    the *largest* size at which every partition's quota still fits inside its own pool
    without replacement: for each partition, ``pool_size * total_weight / weight`` is
    the epoch length at which that partition's quota would exactly exhaust its pool;
    the smallest such value across partitions is the binding constraint (almost always
    the rarest partition relative to its requested share, e.g. ``cancer``). This means
    the rarest partition is seen in full every epoch with no wasted replacement, and
    the majority partition only ever gets a fresh fractional slice — e.g. with pools
    5,000,000 / 70,000 / 30,000 (healthy / healthy_brownish / cancer) and this ratio,
    ``epoch_length`` comes out to 150,000 (quotas 60,000 / 60,000 / 30,000), not
    5,100,000. Pass ``epoch_length`` explicitly to override — a larger value trades
    this guarantee for more repetition of the binding partition within each epoch.
    """

    def __init__(
        self,
        dataset: ConcatDataset[TileEmbeddingClassificationDataset],
        ratio: Mapping[str, float],
        epoch_length: int | None = None,
    ) -> None:
        unknown = ratio.keys() - PARTITION_NAMES.keys()
        if unknown:
            msg = f"Unknown partition name(s) {unknown}; expected {set(PARTITION_NAMES)}"
            raise ValueError(msg)

        # ConcatDataset.datasets is typed as list[Dataset[T]] (the generic base, not T
        # itself) in PyTorch's own stubs — partition_labels() is only on the concrete
        # TileEmbeddingClassificationDataset, which every sub-dataset actually is here.
        sub_datasets = cast(
            "list[TileEmbeddingClassificationDataset]", dataset.datasets
        )
        labels = np.concatenate([sub.partition_labels() for sub in sub_datasets])
        if len(labels) != len(dataset):
            msg = "partition_labels() length mismatch with the dataset it was computed from"
            raise ValueError(msg)

        self._indices_by_code = {
            PARTITION_NAMES[name]: np.flatnonzero(labels == PARTITION_NAMES[name])
            for name in ratio
        }

        total_weight = sum(ratio.values())
        if epoch_length is None:
            # Pools with zero tiles are skipped (not treated as a 0-length
            # constraint) since __iter__ already tolerates an empty partition by
            # drawing nothing from it — one accidentally-empty partition shouldn't
            # collapse the epoch length for every other partition too.
            supportable = [
                len(self._indices_by_code[PARTITION_NAMES[name]]) * total_weight / weight
                for name, weight in ratio.items()
                if len(self._indices_by_code[PARTITION_NAMES[name]]) > 0
            ]
            epoch_length = int(min(supportable)) if supportable else 0

        self._counts = {
            PARTITION_NAMES[name]: max(1, round(epoch_length * weight / total_weight))
            for name, weight in ratio.items()
        }

        # Fixed for the whole run (recomputed identically on every setup() call, not
        # per epoch) — params, not metrics. Safe to call repeatedly: mlflow only
        # errors on logging a *different* value for an existing key, and these are
        # deterministic given the same ratio/pool sizes.
        mlflow.log_params({
            "train_sampler/epoch_length": epoch_length,
            **{
                f"train_sampler/pool_size_{name}": len(self._indices_by_code[PARTITION_NAMES[name]])
                for name in ratio
            },
        })

    def __len__(self) -> int:
        return sum(self._counts.values())

    def __iter__(self) -> Iterator[int]:
        # Sampler.__iter__ runs once per epoch in the main process (DataLoader workers
        # only fetch the indices it yields) — safe to use the global numpy RNG here,
        # already seeded reproducibly by lightning.seed_everything.
        drawn = []
        for code, count in self._counts.items():
            pool = self._indices_by_code[code]
            if len(pool) == 0:
                continue
            # Without replacement when the partition covers this epoch's quota (the
            # common case for "healthy"); with replacement when it doesn't (the common
            # case for "cancer" under severe imbalance) rather than erroring.
            replace = count > len(pool)
            drawn.append(np.random.choice(pool, size=count, replace=replace))

        combined = np.concatenate(drawn)
        np.random.shuffle(combined)
        yield from combined.tolist()

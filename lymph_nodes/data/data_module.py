from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


if TYPE_CHECKING:
    from collections.abc import Iterable

    from lymph_nodes.typing import Metadata, TileEmbeddingsInput


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        kfold_splits: int | None = None,
        k: int | None = None,
        pos_weight: float = 1.0,
        sampler_num_samples: int | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kfold_splits = kfold_splits
        self.k = k
        self.pos_weight = pos_weight
        self.sampler_num_samples = sampler_num_samples
        self.datasets_cfg = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit" | "validate":
                dataset = instantiate(self.datasets_cfg["train"])

                if self.kfold_splits is not None:
                    assert self.k is not None

                    sgkf = StratifiedGroupKFold(
                        n_splits=self.kfold_splits,
                        shuffle=True,
                        random_state=42,
                    )

                    indices = np.arange(len(dataset))
                    splits = list(
                        sgkf.split(
                            indices,
                            dataset.labels,
                            dataset.groups,
                        )
                    )

                    train_idx, val_idx = splits[self.k - 1]
                    
                    self.train = Subset(dataset, train_idx)
                    self.val = Subset(dataset, val_idx)
                else:
                    self.train = dataset
                    self.val = instantiate(self.datasets_cfg["val"])

                self._train_sampler = _build_weighted_sampler(
                    self.train, self.pos_weight, self.sampler_num_samples
                )

            case "test":
                self.test = instantiate(self.datasets_cfg["test"])
            case "predict":
                self.predict = instantiate(self.datasets_cfg["predict"])

    def train_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=self._train_sampler,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def val_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def test_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )


def _build_weighted_sampler(
    dataset: Subset | Any,
    pos_weight: float,
    num_samples: int | None,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples the minority class.

    Per-sample weight = inverse class frequency, with the positive class
    additionally scaled by *pos_weight* (1.0 → 50/50, 0.5 → ~25/75 pos/neg).
    *num_samples* caps the epoch length so validation runs more often.
    """
    if isinstance(dataset, Subset):
        labels = np.asarray(dataset.dataset.labels)[dataset.indices]
    else:
        labels = np.asarray(dataset.labels)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    w_pos = (1.0 / n_pos) * pos_weight if n_pos > 0 else 0.0
    w_neg = 1.0 / n_neg if n_neg > 0 else 0.0

    sample_weights = torch.tensor(
        np.where(labels == 1, w_pos, w_neg),
        dtype=torch.float32,
    )
    del labels

    if num_samples is None:
        num_samples = len(sample_weights)

    return WeightedRandomSampler(
        sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Metadata]],
) -> tuple[Tensor, Tensor, list[Metadata]]:
    inputs, labels, metadatas = zip(*batch, strict=False)
    return torch.stack(inputs), torch.stack(labels), list(metadatas)

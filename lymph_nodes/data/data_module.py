import logging
from collections.abc import Iterable

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from lymph_nodes.data.datasets import (
    DatasetSubset,
    TileEmbeddings,
    TileEmbeddingsSubset,
    create_subset,
)
from lymph_nodes.typing import Metadata, TileEmbeddingsInput


log = logging.getLogger(__name__)


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        kfold_splits: int | None = None,
        k: int | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kfold_splits = kfold_splits
        self.k = k

        if self.kfold_splits is None and self.k is not None:
            raise ValueError("kfold_splits cannot be None if k is set.")

        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit" | "validate":
                if self.kfold_splits is not None:
                    assert self.k is not None
                    dataset = instantiate(self.datasets["train"])
                    sgkf = StratifiedGroupKFold(
                        n_splits=self.kfold_splits, shuffle=True, random_state=42
                    )
                    splits = list(
                        sgkf.split(
                            range(len(dataset)),
                            dataset.labels,
                            dataset.groups,
                        )
                    )
                    train_idx, val_idx = splits[self.k - 1]
                    self.train = create_subset(dataset, train_idx)
                    self.val = create_subset(dataset, val_idx)
                    _log_split(self.train, self.val, fold=self.k)
                else:
                    self.train = instantiate(self.datasets["train"])
                    self.val = instantiate(self.datasets["val"])
                    _log_split(self.train, self.val)
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=_weighted_sampler(self.train),
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
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


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Metadata]],
) -> tuple[Tensor, Tensor, list[Metadata]]:
    inputs = []
    labels = []
    metadatas = []
    for input, label, metadata in batch:
        inputs.append(input)
        labels.append(label)
        metadatas.append(metadata)
    return torch.stack(inputs), torch.stack(labels), metadatas


def _weighted_sampler(
    subset: TileEmbeddings | TileEmbeddingsSubset | DatasetSubset,
) -> WeightedRandomSampler:
    """Create a weighted random sampler to balance positive/negative classes."""
    labels = subset.labels
    class_counts = {c: labels.count(c) for c in set(labels)}
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def _log_split(
    train: TileEmbeddings | TileEmbeddingsSubset | DatasetSubset,
    val: TileEmbeddings | TileEmbeddingsSubset | DatasetSubset,
    fold: int | None = None,
) -> None:
    """Log a human-readable summary of the train/val split to stdout."""
    prefix = f"Fold {fold} — " if fold is not None else ""

    def _summarise(
        subset: TileEmbeddings | TileEmbeddingsSubset | DatasetSubset, name: str
    ) -> None:
        pos = sum(subset.labels)
        neg = len(subset.labels) - pos
        log.info(
            "%s%s  (%d slides: %d pos, %d neg)",
            prefix,
            name,
            len(subset.labels),
            pos,
            neg,
        )
        for slide in subset.slides:
            marker = "+" if slide["label"] == 1 else "-"
            log.info("    [%s] %s", marker, slide["name"])

    log.info("=" * 60)
    _summarise(train, "TRAIN")
    log.info("-" * 60)
    _summarise(val, "VAL")
    log.info("=" * 60)

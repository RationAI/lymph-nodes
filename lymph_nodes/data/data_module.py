from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor
from torch.utils.data import DataLoader, Subset


if TYPE_CHECKING:
    from collections.abc import Iterable

    from lymph_nodes.typing import Metadata, TileEmbeddingsInput


log = logging.getLogger(__name__)


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        kfold_splits: int | None = None,
        k: int | None = None,
        pos_weight: float = 1.0,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kfold_splits = kfold_splits
        self.k = k
        self.pos_weight = pos_weight
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

                    _log_split(self.train, self.val, fold=self.k)
                else:
                    self.train = dataset
                    self.val = instantiate(self.datasets_cfg["val"])
                    _log_split(self.train, self.val)

            case "test":
                self.test = instantiate(self.datasets_cfg["test"])
            case "predict":
                self.predict = instantiate(self.datasets_cfg["predict"])

    def train_dataloader(self) -> Iterable[TileEmbeddingsInput]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            # sampler=_weighted_sampler(self.train, self.pos_weight),
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


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Metadata]],
) -> tuple[Tensor, Tensor, list[Metadata]]:
    inputs, labels, metadatas = zip(*batch, strict=False)
    return torch.stack(inputs), torch.stack(labels), list(metadatas)


def _log_split(
    train: Subset | Any,
    val: Subset | Any,
    fold: int | None = None,
) -> None:
    """Logs class and slide distribution for the current split."""
    prefix = f"Fold {fold} — " if fold is not None else ""

    def _summarise(subset: Subset | Any, name: str) -> None:
        # Resolve dataset if it's a Subset
        base_ds = subset.dataset if isinstance(subset, Subset) else subset
        indices = (
            subset.indices if isinstance(subset, Subset) else np.arange(len(subset))
        )

        current_labels = base_ds.labels[indices]
        n_samples = len(current_labels)

        # Slides are handled by the custom 'slides' property in MLPEmbeddingDataset
        # We filter to only show slides present in this subset
        if isinstance(subset, Subset):
            current_slide_ids = set(base_ds.groups[indices])
            subset_slides = [
                s for s in base_ds.slides if s["name"] in current_slide_ids
            ]
        else:
            subset_slides = base_ds.slides

        n_slides = len(subset_slides)
        pos_slides = sum(s["label"] for s in subset_slides)
        neg_slides = n_slides - pos_slides

        log.info("=" * 60)
        log.info(
            "%s%s  (%d tiles from %d slides: %d+ / %d- slides)",
            prefix,
            name,
            n_samples,
            n_slides,
            pos_slides,
            neg_slides,
        )

        # Log individual slide status
        for slide in subset_slides:
            marker = "+" if slide["label"] == 1 else "-"
            log.info("    [%s] %s", marker, slide["name"])

    _summarise(train, "TRAIN")
    log.info("-" * 60)
    _summarise(val, "VAL")
    log.info("=" * 60)

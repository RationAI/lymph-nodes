from collections.abc import Iterable

import numpy as np
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from rationai.mlkit.data.samplers import PDMulticlassBatchSampler
from torch.utils.data import DataLoader

from lymph_nodes.typing import Sample


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        epoch_size: int | None,
        positive_batch_split: float,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.positive_batch_split = positive_batch_split
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.val = instantiate(self.datasets["val"])
            case "validate":
                self.val = instantiate(self.datasets["val"])
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[Sample]:
        return DataLoader(
            self.train,
            batch_sampler=PDMulticlassBatchSampler(
                self.train.tiles,
                stratify_by="cancer",
                distribution=np.array(
                    [1 - self.positive_batch_split, self.positive_batch_split]
                ),
                batch_size=self.batch_size,
                epoch_size=self.epoch_size,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Sample]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> list[Iterable[Sample]]:
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            for dataset in self.test.datasets
        ]

    def predict_dataloader(self) -> list[Iterable[Sample]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]

from collections.abc import Iterable

import numpy as np
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from rationai.mlkit.data.samplers import PDMulticlassBatchSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lymph_nodes.typing import Input


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        epoch_size: int | None,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit" | "validate":
                self.train_val_dataset = instantiate(self.datasets["train"])
                self.train_indices, self.val_indices = train_test_split(
                    self.train_val_dataset.tiles.index,
                    test_size=0.1,
                    stratify=self.train_val_dataset.tiles["metastazis"],
                )
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.train_val_dataset,
            batch_sampler=PDMulticlassBatchSampler(
                self.train_val_dataset.tiles.loc[self.train_indices],
                stratify_by="metastazis",
                distribution=np.array([0.8, 0.2]),
                batch_size=self.batch_size,
                epoch_size=self.epoch_size,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.train_val_dataset,
            batch_sampler=PDMulticlassBatchSampler(
                self.train_val_dataset.tiles.loc[self.val_indices],
                stratify_by="metastazis",
                distribution=np.array([0.8, 0.2]),
                batch_size=self.batch_size,
                epoch_size=self.epoch_size // 100,
            ),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> list[Iterable[Input]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]

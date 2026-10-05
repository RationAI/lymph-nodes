from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from lymph_nodes.typedefs import Batch


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        train_sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        self._train_sampler_cfg = train_sampler
        self.train_sampler = None

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.val = instantiate(self.datasets["val"])
                # Needs the real train dataset (to read partition_labels() off its
                # per-slide sub-datasets), so it can only be built after self.train —
                # unlike train/val/test, which only need their own static config.
                if self._train_sampler_cfg is not None:
                    self.train_sampler = instantiate(self._train_sampler_cfg, dataset=self.train)
            case "validate":
                self.val = instantiate(self.datasets["eval"])
            case "test":
                self.test = instantiate(self.datasets["test"])

    def train_dataloader(self) -> Iterable[Batch]:
        if self.train_sampler is not None:
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

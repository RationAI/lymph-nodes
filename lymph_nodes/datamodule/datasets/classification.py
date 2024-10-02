# Copyright (c) The RationAI team
import random
from pathlib import Path

import albumentations
import mlflow
import torch

from lymph_nodes.datamodule.datasets.base_wsi import BaseDataset, extract_tile
from lymph_nodes.datamodule.samplers import BaseSampler


class ClassificationDataset(BaseDataset):
    transforms: albumentations.TemplateTransform | None
    _downloaded: dict[str, str]  # mapping URI -> fp

    def __init__(
        self,
        sampler: BaseSampler,
        seed: int,
        augmentations: albumentations.TemplateTransform | None = None,
    ) -> None:
        super().__init__(sampler=sampler, seed=seed)
        self.transforms = augmentations
        self._downloaded: dict[str, str] = {}

    def __it_is_mlflow_uri(self, uri):
        if not isinstance(uri, str):
            raise TypeError(f"Data URI must be a string, got {type(uri)}")
        if not uri.startswith("mlflow-artifacts:/"):
            return False
        return True

    def __download_sample_artifacts(self, uri: str) -> str:
        """Downloads artifacts from mlflow URI and stores them in `./slides/` directory.

        Avoids repeated downloads by caching paths of downloaded artifacts in `_downloaded` dict.
        """
        if uri in self._downloaded:
            slide_fp = self._downloaded[uri]
        else:
            slide_fp = mlflow.artifacts.download_artifacts(
                artifact_uri=uri, dst_path="./slides/"
            )
            self._downloaded[uri] = slide_fp
        return slide_fp

    def generate_samples(self) -> None:
        """This method gathers samples from sampler and also downloads all slides for current epoch.

        It modifies samples: adds `slide_fp` keys.
        """
        super().generate_samples()

        # Downloading slides and masks from mlflow URI and storing fp in sample
        for sample in self._epoch_samples:
            if self.__it_is_mlflow_uri(sample["slide_fp"]):
                sample["slide_fp"] = self.__download_sample_artifacts(
                    uri=sample["mlflow_slide_uri"]
                )
            uri = sample["mlflow_slide_uri"] = 0  # noqa: F841

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        sample = self._epoch_samples[index]

        image = extract_tile(
            slide_fp=Path(sample.get("slide_fp")).resolve(),
            coord_x=sample["coord_x"],
            coord_y=sample["coord_y"],
            tile_size=sample["tile_size"],
            level=sample["sample_level"],
        )

        if self.transforms:
            random.seed(int(self._rng.integers(0, 2**63 - 1)))
            image = self.transforms(image=image)["image"]

        # permute to (channels, height, width)
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.FloatTensor([sample["class_id"]])
        return image, label, sample

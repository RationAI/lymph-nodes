from pathlib import Path
from typing import Any, TypedDict, cast

import mlflow
import pandas as pd
import torch
from lightning import Callback, Trainer
from rationai.masks.mask_builders import ScalarMaskBuilder


class MaskBuilders(TypedDict):
    attention_rescaled: ScalarMaskBuilder
    classification_positive: ScalarMaskBuilder
    classification_negative: ScalarMaskBuilder


def min_max_normalization(tensor: torch.Tensor) -> torch.Tensor:
    values_max = tensor.max()
    values_min = tensor.min()
    denominator = (values_max - values_min).clamp_min(1e-12)
    return (tensor - values_min) / denominator


def sigmoid_normalization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(tensor)


class MaskBuilderCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._mask_builders: dict[str, MaskBuilders] = {}

    def _resolve_slides(self, trainer: Trainer) -> pd.DataFrame:
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("Trainer has no datamodule attached.")

        candidates = [
            getattr(datamodule, "predict", None),
            getattr(datamodule, "test", None),
            getattr(datamodule, "val", None),
            getattr(datamodule, "train", None),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            slides = getattr(candidate, "slides", None)
            if slides is None:
                continue
            if isinstance(slides, pd.DataFrame):
                dataframe = slides.copy()
            else:
                dataframe = pd.DataFrame(slides)

            if "path" not in dataframe.columns:
                continue

            dataframe["name"] = dataframe["path"].apply(lambda value: Path(value).stem)
            return dataframe

        raise RuntimeError(
            "Unable to locate slides metadata with a 'path' column on datamodule datasets."
        )

    def _get_mask_builders(self, slide_name: str, trainer: Trainer) -> MaskBuilders:
        if slide_name in self._mask_builders:
            return self._mask_builders[slide_name]

        slides = self._resolve_slides(trainer)
        slide_rows = slides[slides["name"] == slide_name]
        if slide_rows.empty:
            raise ValueError(f"Slide '{slide_name}' was not found in dataset metadata.")

        slide = cast(pd.Series, slide_rows.iloc[0])

        kwargs = {
            "filename": Path(str(slide["path"])).stem,
            "extent_x": int(slide["extent_x"]),
            "extent_y": int(slide["extent_y"]),
            "mpp_x": float(slide["mpp_x"]),
            "mpp_y": float(slide["mpp_y"]),
            "extent_tile": int(slide["tile_extent_x"]),
            "stride": int(slide["stride_x"]),
        }

        builders: MaskBuilders = {
            "attention_rescaled": ScalarMaskBuilder(
                save_dir=Path("masks/attention_rescaled"),
                **kwargs,
            ),
            "classification_positive": ScalarMaskBuilder(
                save_dir=Path("masks/classification_positive"),
                **kwargs,
            ),
            "classification_negative": ScalarMaskBuilder(
                save_dir=Path("masks/classification_negative"),
                **kwargs,
            ),
        }

        self._mask_builders[slide_name] = builders
        return builders

    def _save_mask_builders(self) -> None:
        for per_slide_builders in self._mask_builders.values():
            for mask_builder in per_slide_builders.values():
                artifact = mask_builder.save()
                mlflow.log_artifact(
                    str(artifact),
                    artifact_path=str(mask_builder.save_dir),
                )

    def _extract_batch_items(self, batch: Any) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            return []

        bags, metadata_list = batch[0], batch[1]
        if not isinstance(metadata_list, (tuple, list)):
            return []

        pairs: list[tuple[torch.Tensor, dict[str, Any]]] = []
        for bag, metadata in zip(bags, metadata_list, strict=True):
            if isinstance(bag, torch.Tensor) and isinstance(metadata, dict):
                pairs.append((bag, metadata))
        return pairs

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _ = outputs, batch_idx, dataloader_idx

        for bag, metadata in self._extract_batch_items(batch):
            slide_name = str(metadata.get("slide_name") or metadata.get("name") or "")
            if not slide_name:
                continue

            x_coords = metadata.get("x") or metadata.get("tile_x")
            y_coords = metadata.get("y") or metadata.get("tile_y")
            if x_coords is None or y_coords is None:
                continue

            item_count = min(len(x_coords), len(y_coords))
            if item_count == 0:
                continue

            mask_builders = self._get_mask_builders(slide_name, trainer)
            bag = bag[:item_count]

            encoded = pl_module.encoder(bag)
            attention_weights = sigmoid_normalization(pl_module.attention(encoded)).cpu()
            attention_weights = attention_weights.squeeze(-1)

            mask_builders["attention_rescaled"].update(
                min_max_normalization(attention_weights),
                x_coords,
                y_coords,
            )

            classification = pl_module.classifier(encoded).cpu()
            if classification.ndim == 1 or (classification.ndim == 2 and classification.shape[-1] == 1):
                positive_probability = classification.sigmoid().reshape(-1)
                negative_probability = 1.0 - positive_probability
                mask_builders["classification_positive"].update(
                    positive_probability,
                    x_coords,
                    y_coords,
                )
                mask_builders["classification_negative"].update(
                    negative_probability,
                    x_coords,
                    y_coords,
                )
            elif classification.ndim == 2 and classification.shape[-1] == 2:
                probabilities = torch.softmax(classification, dim=-1)
                negative_probability = probabilities[:, 0]
                positive_probability = probabilities[:, 1]
                mask_builders["classification_positive"].update(
                    positive_probability,
                    x_coords,
                    y_coords,
                )
                mask_builders["classification_negative"].update(
                    negative_probability,
                    x_coords,
                    y_coords,
                )

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
        _ = trainer, pl_module
        if not self._mask_builders:
            return
        self._save_mask_builders()
        self._mask_builders.clear()
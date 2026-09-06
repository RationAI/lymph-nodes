from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch


class FoundationModelEncoder(ABC):
    """Base class for ViT foundation model inference via Ray map_batches.

    Subclasses implement _create_model() and optionally _build_embedding_fn()
    to define what gets compiled and called during inference.

    Shared optimisations:
    - GPU normalisation: numpy stack → tensor in one operation, no PIL loop
    - torch.compile(reduce-overhead): JIT-compiles the forward function
    - autocast bf16/fp16: mixed precision; Flash Attention via PyTorch SDPA
    - cudnn.benchmark: cuDNN auto-selects optimal kernels for fixed input size
    """

    def __init__(self, image_col: str, embedding_col: str) -> None:
        from timm.data import resolve_model_data_config

        self._image_col = image_col
        self._embedding_col = embedding_col

        torch.backends.cudnn.benchmark = True
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        model = self._create_model()
        model.eval().to(self.device)

        cfg = resolve_model_data_config(model)
        self.mean: torch.Tensor = torch.tensor(cfg["mean"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std: torch.Tensor = torch.tensor(cfg["std"], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.autocast_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self._compiled: Callable[[torch.Tensor], torch.Tensor] = torch.compile(
            self._build_embedding_fn(model), mode="max-autotune"
        )

    @abstractmethod
    def _create_model(self) -> torch.nn.Module: ...

    def _build_embedding_fn(self, model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a callable that maps images → embeddings.

        Default: the model itself (standard forward pass).
        Override for models that require non-standard embedding extraction.
        """
        return model

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = (
            torch.from_numpy(np.ascontiguousarray(np.stack(batch[self._image_col])))
            .to(self.device)
            .permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
            .float()
            .div_(255.0)
            .sub_(self.mean)
            .div_(self.std)
        )

        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
            embeddings = self._compiled(images)

        batch[self._embedding_col] = embeddings.float().cpu().numpy()  # (B, D) float32
        return batch


class GigaPathEncoder(FoundationModelEncoder):
    """Microsoft GigaPath — output: 1536-dim embedding.

    https://huggingface.co/prov-gigapath/prov-gigapath
    """

    def _create_model(self) -> torch.nn.Module:
        import timm
        return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


class UNI2Encoder(FoundationModelEncoder):
    """MahmoodLab UNI2-h — output: 1536-dim embedding.

    https://huggingface.co/MahmoodLab/UNI2-h
    """

    def _create_model(self) -> torch.nn.Module:
        import timm
        return timm.create_model(
            "hf_hub:MahmoodLab/uni2-h",
            pretrained=True,
            init_values=1e-5,
        )


class Virchow2Encoder(FoundationModelEncoder):
    """Paige Virchow2 — output: 2560-dim embedding (CLS token + mean patch tokens).

    https://huggingface.co/paige-ai/Virchow2
    """

    def _create_model(self) -> torch.nn.Module:
        import timm
        return timm.create_model(
            "hf_hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            # Pass an activation factory explicitly; passing an activation
            # tensor here causes timm to fail when constructing the MLP.
            act_layer=lambda **kwargs: torch.nn.SiLU(**kwargs),
        )

    def _build_embedding_fn(self, model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
        # Virchow2 embedding = CLS token ‖ mean(patch tokens), giving 2 x 1280 = 2560 dims.
        # Compiles the whole extraction as one function so torch.compile sees
        # forward_features + concat as a single traced graph.
        def embed(images: torch.Tensor) -> torch.Tensor:
            tokens = model.forward_features(images)   # (B, 1 + n_patches, 1280)
            cls = tokens[:, 0]                         # (B, 1280)
            patches = tokens[:, 1:].mean(dim=1)        # (B, 1280)
            return torch.cat([cls, patches], dim=-1)   # (B, 2560)

        return embed

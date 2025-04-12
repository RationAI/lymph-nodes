from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pyvips
import torch
from numpy.typing import NDArray
from rationai.masks import write_big_tiff
from rationai.masks.mask_builders.mask_builder import MaskBuilder as BaseMaskBuilder


class MaskBuilder(BaseMaskBuilder, ABC):
    def __init__(
        self,
        save_dir: Path | str,
        filename: str,
        extent_x: int,
        extent_y: int,
        mpp_x: float,
        mpp_y: float,
        tile_extent_x: int,
        tile_extent_y: int,
    ) -> None:
        super().__init__(save_dir, filename, extent_x, extent_y, mpp_x, mpp_y)

        self.image = np.memmap(
            str(self.filename) + "_mask.nmp",
            dtype=np.float32,
            mode="w+",
            shape=(self.extent_y, self.extent_x),
        )

    @abstractmethod
    def update_tile(
        self, tile: NDArray, x: int, y: int, mm: tuple[int, int]
    ) -> None: ...

    def update(self, data: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> None:
        tiles = data.detach().cpu().numpy()
        xs_np = xs.detach().cpu().numpy()
        ys_np = ys.detach().cpu().numpy()

        for tile, x, y in zip(tiles, xs_np, ys_np, strict=True):
            mm = self.image[y : y + tile.shape[0], x : x + tile.shape[1]].shape
            self.update_tile(tile, x, y, mm)

        self.image.flush()

    def save(self) -> Path:
        image_vips = pyvips.Image.new_from_array(self.image)

        image_vips *= 255
        image_vips = image_vips.cast(pyvips.BandFormat.UCHAR)

        path = self.filename.with_suffix(".tiff")
        write_big_tiff(image_vips, path, self.mpp_x, self.mpp_y)
        return path

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lymph_nodes.callbacks.builders.mask_builder import MaskBuilder


class AverageMaskBuilder(MaskBuilder):
    def __init__(
        self,
        save_dir: Path | str,
        filename: str,
        extent_x: int,
        extent_y: int,
        mpp_x: float,
        mpp_y: float,
        tile_extent_x: int = 512,
        tile_extent_y: int = 512,
    ) -> None:
        super().__init__(
            save_dir,
            filename,
            extent_x,
            extent_y,
            mpp_x,
            mpp_y,
        )

        self.kernel = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, tile_extent_y // 2),
                    np.linspace(1, 0, tile_extent_y // 2),
                ]
            ),
            np.concatenate(
                [
                    np.linspace(0, 1, tile_extent_x // 2),
                    np.linspace(1, 0, tile_extent_x // 2),
                ]
            ),
        )

    def update_tile(self, tile: NDArray, x: int, y: int, mm: tuple[int, int]) -> None:
        mm_y, mm_x = mm

        self.image[y : y + mm_y, x : x + mm_x] += (
            self.kernel[:mm_y, :mm_x] * tile[:mm_y, :mm_x]
        )

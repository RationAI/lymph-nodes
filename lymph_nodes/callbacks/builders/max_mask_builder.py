import numpy as np
from numpy.typing import NDArray

from lymph_nodes.callbacks.builders.mask_builder import MaskBuilder


class MaxMaskBuilder(MaskBuilder):
    def update_tile(self, tile: NDArray, x: int, y: int, mm: tuple[int, int]) -> None:
        mm_y, mm_x = mm

        self.image[y : y + mm_y, x : x + mm_x] = np.maximum(
            tile[:mm_y, :mm_x], self.image[y : y + mm_y, x : x + mm_x]
        )

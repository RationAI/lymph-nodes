# Copyright (c) The RationAI team
from pathlib import Path

import numpy as np
import PIL
import slidelip
from nptyping import NDArray


def extract_tile(
    slide_fp: Path, coord_x: int, coord_y: int, tile_size: int, level: int
) -> NDArray:
    """Extracts a tile from a slide using the supplied coordinate values.

    Args:
        slide_fp (Path): path to the slide.
        coord_x (int): coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        coord_y (int): coordinates of a tile to be extracted at OpenSlide level 0 resolution.
        tile_size (int): Size of the tile to be extracted.
        level (int): Resolution level from which tile should be extracted.

    Returns:
        NDArray: RGB Tile represented as numpy array.
    """
    wsi = slidelip.open_slide(slide_fp, slidelip.SlideType.GENERIC)
    bg_tile = PIL.Image.new(mode="RGB", size=(tile_size, tile_size), color="#FFFFFF")
    im_tile = wsi.read_region(
        location=(coord_x, coord_y), level=level, size=(tile_size, tile_size)
    )
    bg_tile.paste(im=im_tile, mask=im_tile, box=None)
    wsi.close()
    return np.array(bg_tile)

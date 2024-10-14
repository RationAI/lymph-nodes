from pathlib import Path

import pyvips
import ray
from openslide import OpenSlide
from rationai.masks import (
    process_items,
    tissue_mask,
    write_big_tiff,
)

from preprocessing.utils import get_level_by_mpp, get_mpp, mpp_to_ppmm


def slide_tissue_mask(slide_path: Path, tissue_mask_mpp: float, dest_dir: Path) -> None:
    with OpenSlide(slide_path) as slide:
        level = get_level_by_mpp(slide, mpp=tissue_mask_mpp)
        xres, yres = mpp_to_ppmm(get_mpp(slide, level))
    slide = pyvips.Image.new_from_file(slide_path, page=level)

    mask = tissue_mask(slide)

    mask_path = Path(dest_dir, f"{Path(slide_path).stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    write_big_tiff(mask, path=mask_path, xres=xres, yres=yres)


def get_tissue_masks(slide_paths: list[Path]) -> None:
    tissue_mask_mpp = 2

    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path("data/tissue_masks", slide_path.parent.stem)  # keep last level
        slide_tissue_mask(slide_path, tissue_mask_mpp, dest_dir)

    process_items(slide_paths, process_item=process_slide)

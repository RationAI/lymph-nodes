from pathlib import Path

import numpy as np
import ray
from openslide import OpenSlide
from PIL import Image
from rationai.masks import (
    process_items,
)

from preprocessing.tissue_mask_new import tissue_mask_new
from preprocessing.utils import get_level_by_mpp, get_mpp, mpp_to_ppmm, vips_read


def slide_tissue_mask(
    slide_path: Path, tissue_mask_mpp: float, dest_dir: Path, d_size: int
) -> None:
    with OpenSlide(slide_path) as slide:
        level = get_level_by_mpp(slide, mpp=tissue_mask_mpp)
        mpp = get_mpp(slide, level)
        xres, yres = mpp_to_ppmm(get_mpp(slide, level))

    slide = vips_read(slide_path, level)

    disk_size = int(d_size // mpp[0])

    mask = tissue_mask_new(slide, disk_size).numpy()
    img = slide.numpy()
    background = Image.fromarray(img).convert("RGBA")
    pil_mask = Image.fromarray(np.clip(mask, 0, 100))

    green_highlight = Image.new("RGBA", pil_mask.size, (0, 255, 0, 0))
    green_highlight.putalpha(pil_mask)

    mask_path = Path(dest_dir, f"{Path(slide_path).stem}_{d_size}-g.png")
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    Image.alpha_composite(background, green_highlight).convert("RGB").save(mask_path)


def find_disc_size(slide_paths: list[Path]) -> None:
    tissue_mask_mpp = 2

    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path(
            "data/tissue_masks_png", slide_path.parent.stem
        )  # keep last level
        slide_tissue_mask(slide_path, tissue_mask_mpp, dest_dir, 10)
        slide_tissue_mask(slide_path, tissue_mask_mpp, dest_dir, 2)

    process_items(slide_paths, process_item=process_slide)

from collections.abc import Iterable
from pathlib import Path

import mlflow
import ray
from openslide import OpenSlide
from rationai.masks import (
    closest_level,
    process_items,
    slide_resolution,
    write_big_tiff,
)

from prepro.algorithms import tissue_mask
from prepro.utils import get_relative_dir_path, vips_read


def slide_tissue_mask(slide_path: Path, desired_mpp: float, dest_dir: Path) -> None:
    with OpenSlide(slide_path) as slide:
        level = closest_level(slide, desired_mpp)
        mpp_x, mpp_y = slide_resolution(slide, level)

    slide = vips_read(slide_path, level)

    mask = tissue_mask(slide, disk_size=int(10 // mpp_x))

    mask_path = Path(dest_dir, f"{Path(slide_path).stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


def generate_tissue_masks(
    slide_paths: Iterable[Path], mpp: float, reference_path: str, dest: str
) -> None:
    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path(
            dest, get_relative_dir_path(slide_path, Path(reference_path))
        )  # keep last level
        slide_tissue_mask(slide_path, mpp, dest_dir)

    process_items(slide_paths, process_item=process_slide)

    mlflow.log_artifacts(dest, artifact_path="tissue_masks")

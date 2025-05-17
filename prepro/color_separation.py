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
from rationai.masks.vips_filters import VipsClosing, VipsCompose, VipsOpening

from prepro.utils import vips_read


DISC_FACTOR = 5

morph_filters = VipsCompose(
    [
        VipsOpening(DISC_FACTOR),
        VipsClosing(DISC_FACTOR),
    ]
)


def slide_color_separation(
    slide_path: Path, desired_mpp: float, dest_dir: Path
) -> None:
    with OpenSlide(slide_path) as slide:
        level = closest_level(slide, desired_mpp)
        mpp = slide_resolution(slide, level)
    slide = vips_read(slide_path, level=level)

    red, green, blue, *_ = slide.bandsplit()

    mask = (
        (red + green > blue + 50)
        & (red > green)
        & (green >= blue - 10)
        & (red >= 50)
        & (red < 175)
    )

    mask, (mpp_x, mpp_y) = morph_filters(mask, mpp)

    mask_path = Path(dest_dir, f"{Path(slide_path).stem}.tiff")
    mask_path.parent.mkdir(exist_ok=True, parents=True)

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


def color_separation(
    slides: Iterable[Path], mpp: float, reference_path: str, dest: str
) -> None:
    @ray.remote
    def process_slide(slide_path: Path) -> None:
        dest_dir = Path(dest, slide_path.relative_to(reference_path).parent)
        slide_color_separation(slide_path, mpp, dest_dir)

    process_items(slides, process_slide)


def generate_color_separation_masks(
    slide_paths: Iterable[Path], mpp: float, reference_path: str, dest: str
) -> None:
    color_separation(slide_paths, mpp, reference_path, dest)
    mlflow.log_artifacts(dest, artifact_path="color_separation_masks")


if __name__ == "__main__":
    # Testing setup (to define suitable thresholds for brownish (eg metastazis) color separation)

    TEST_MPP = 2

    # Testing WSIs
    test_wsis = [
        # Annotated image
        "/mnt/data/Projects/lymph_nodes/annotated_ihc_test/SNB_IHC_TEST_CASE-2024_1011-15.mrxs",
        # Positive samples
        "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023/SNB_IHC_CASE_1_SLIDE_3-1.tiff",
        "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023/SNB_IHC_CASE_2_SLIDE_1-1.tiff",
        # negative samples
        "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023/SNB_IHC_CASE_3_SLIDE_1-0.tiff",
        "/mnt/data/Projects/lymph_nodes/dataset1-ihc-2023/SNB_IHC_CASE_4_SLIDE_3-0.mrxs",
    ]

    reference_path = "/mnt/data/Projects/lymph_nodes"
    destination = "/mnt/data/Projects/lymph_nodes/dev/baseline-v2/color_separation"

    color_separation(list(map(Path, test_wsis)), TEST_MPP, reference_path, destination)

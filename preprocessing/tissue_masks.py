from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import hydra
import pyvips
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.openslide import OpenSlide


####################################################################################
#      This implementaion differs from traditional tissue_mask used for H&E        #
# slides beacuse it is not designed for HDAB slides which are used in this project #
####################################################################################


def tissue_mask(slide: pyvips.Image, disk_size: int = 10) -> pyvips.Image:
    """Generates a tissue mask from a whole-slide image (WSI) using saturation channel extraction and morphological operations, and saves the mask as a TIFF image.

    The function extracts the saturation channel from the WSI, to identify tissue regions,
    and performs morphological operations (closing and opening) to refine the mask.

    Args:
        slide: whole-slide image (WSI) pyvips file handler.
        disk_size: Size of the disk element for morphological operations (default is 10).

    Returns:
        The generated tissue mask as pyvips.Image.
    """
    # Extract saturation channel
    vi_slide_hsv = slide.sRGB2HSV()
    _, vi_s, vi_v, *_ = vi_slide_hsv.bandsplit()

    tresholded = (vi_v > 50) & ((vi_v < 240) | (vi_s > 10))

    # Morph Object
    vi_disk_object = pyvips.Image.black(2 * disk_size + 1, 2 * disk_size + 1) + 128
    vi_disk_object = vi_disk_object.draw_circle(
        255, disk_size, disk_size, disk_size, fill=True
    )

    # Closing
    vi_mask = tresholded.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)

    # Opening
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)

    return vi_mask


@ray.remote(memory=3 * 1024**3)
def process_slide(slide_path: str, mpp: int, output_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        mpp_x, mpp_y = slide.slide_resolution(level)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))

    mask = tissue_mask(slide, disk_size=int(10 // (mpp_x + mpp_y) / 2))

    mask_path = output_path / Path(slide_path).with_suffix(".tiff").name
    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


@with_cli_args(["+preprocessing=tissue_masks"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = hydra.utils.instantiate(config.dataset.slides)

    with TemporaryDirectory() as output_dir:
        process_items(
            slides,
            process_item=process_slide,
            fn_kwargs={
                "mpp": config.mpp,
                "output_path": Path(output_dir),
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(local_dir=output_dir, artifact_path=config.artifact_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

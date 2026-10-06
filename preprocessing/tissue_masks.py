from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import hydra
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.openslide import OpenSlide


####################################################################################
#      This implementation differs from the traditional tissue_mask used for H&E   #
#      slides because that one is not designed for the HDAB slides used here.      #
####################################################################################


def tissue_mask(slide: pyvips.Image, disk_size: int = 10) -> pyvips.Image:
    """Generates a tissue mask from a whole-slide image (WSI) via HSV thresholding and morphology.

    A pixel counts as tissue when it is not too dark (V > 50) and either not near-white
    (V < 240) or noticeably saturated (S > 10). The result is refined by a morphological
    closing followed by an opening.

    Args:
        slide: whole-slide image (WSI) pyvips file handler.
        disk_size: Radius in pixels of the disk element for the morphological operations.

    Returns:
        The generated tissue mask as pyvips.Image.
    """
    # Extract saturation and value channels
    vi_slide_hsv = slide.sRGB2HSV()
    _, vi_s, vi_v, *_ = vi_slide_hsv.bandsplit()

    thresholded = (vi_v > 50) & ((vi_v < 240) | (vi_s > 10))

    # Morph Object
    vi_disk_object = pyvips.Image.black(2 * disk_size + 1, 2 * disk_size + 1) + 128
    vi_disk_object = vi_disk_object.draw_circle(
        255, disk_size, disk_size, disk_size, fill=True
    )

    # Closing
    vi_mask = thresholded.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)

    # Opening
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)
    vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)

    return vi_mask


@ray.remote(memory=3 * 1024**3)
def process_slide(slide_path: str, mpp: float, output_path: Path) -> None:
    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        mpp_x, mpp_y = slide.slide_resolution(level)

    slide = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path, level=level))

    # Disk radius of 10 µm, expressed in pixels at the mask's resolution.
    mask = tissue_mask(slide, disk_size=max(1, round(10 / ((mpp_x + mpp_y) / 2))))

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

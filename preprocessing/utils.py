import numpy as np
from openslide import PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, OpenSlide


def mpp_to_ppmm(mpp: tuple[float, float]) -> tuple[float, float]:
    """Convert microns per pixel to pixels per millimeter."""
    return 1000 / mpp[0], 1000 / mpp[1]


def get_mpp(slide: OpenSlide, level: int) -> tuple[float, float]:
    """Get microns per pixel for the given slide and level."""
    return (
        float(slide.properties[PROPERTY_NAME_MPP_X]) * slide.level_downsamples[level],
        float(slide.properties[PROPERTY_NAME_MPP_Y]) * slide.level_downsamples[level],
    )


def get_level_by_mpp(slide: OpenSlide, mpp: float | tuple[float, float]) -> int:
    """Get the level with the highest microns per pixel."""
    slide_mpp = np.array(
        [
            float(slide.properties[PROPERTY_NAME_MPP_X]),
            float(slide.properties[PROPERTY_NAME_MPP_Y]),
        ]
    )
    scale_factor = np.average(mpp / slide_mpp)

    return np.abs(np.asarray(slide.level_downsamples) - scale_factor).argmin().item()

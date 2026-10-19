"""Per-slide white point: correcting scans whose empty glass is not white.

Some scanners are poorly calibrated: the FN Brno scans (Zeiss Axio Scan.Z1) show the
empty glass at RGB ~185-205 instead of ~255, so all their tissue looks dark and much of
it falls into the colour range of DAB (preprocessing/tiling_blocks/brownish.py) and
below the background threshold of the tissue detector. The slide's own glass is a
reference of the scanner's white: scaling each channel so that it becomes 255
(``apply_white_point``) brings such scans close to well calibrated ones and leaves
those almost unchanged (glass at ~252-254).

The white point is estimated once per slide (``estimate_white_point``), stored in the
slides table (``white_r``, ``white_g``, ``white_b``; preprocessing.data_exploration.
patient_dataset) and applied wherever pixels are read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Mapping

    from openslide import OpenSlide


WHITE_POINT_COLUMNS = ("white_r", "white_g", "white_b")


def estimate_white_point(
    slide: OpenSlide,
    target_mpp: float = 8.0,
    max_saturation: float = 0.10,
    brightest_fraction: float = 0.2,
) -> tuple[float, float, float]:
    """RGB of the slide's empty glass, from a low-resolution level.

    The glass is taken as the brightest ``brightest_fraction`` of the low-saturation
    pixels of the scanned area. Unscanned regions are excluded: transparent ones, and
    the opaque pure white or black that some formats fill them with (MRXS, CZI through
    the OpenSlide fork) — real glass is never exactly (255, 255, 255). Returns
    (255, 255, 255) if the slide has no such pixels.
    """
    mpp = float(slide.properties.get("openslide.mpp-x", 0.25))
    level = slide.get_best_level_for_downsample(target_mpp / mpp)
    rgba = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]))
    rgb = rgba[..., :3].astype(np.float32)
    total = rgb.sum(axis=-1)
    scanned = (rgba[..., 3] > 0) & (total > 0) & (total < 3 * 255)

    high, low = rgb.max(axis=-1), rgb.min(axis=-1)
    saturation = (high - low) / np.maximum(high, 1)
    candidates = scanned & (saturation < max_saturation)
    if not candidates.any():
        return (255.0, 255.0, 255.0)
    lightness = rgb.mean(axis=-1)
    threshold = np.quantile(lightness[candidates], 1 - brightest_fraction)
    glass = rgb[candidates & (lightness >= threshold)]
    red, green, blue = np.median(glass, axis=0)
    return (float(red), float(green), float(blue))


def white_point_of(row: Mapping[str, object]) -> np.ndarray | None:
    """The white point stored in a slides-table row, or None if it has none."""
    values = [row.get(column) for column in WHITE_POINT_COLUMNS]
    if any(value is None or value != value for value in values):  # missing or NaN
        return None
    return np.asarray(values, dtype=np.float32)


def apply_white_point(
    rgb: np.ndarray, white: np.ndarray | None, min_white: float = 100.0
) -> np.ndarray:
    """``rgb`` (..., 3, uint8) with each channel scaled so that ``white`` becomes 255.

    ``white`` None leaves the image unchanged; channels are clamped at ``min_white`` so
    that a bad estimate cannot blow the image up.
    """
    if white is None:
        return rgb
    gain = 255.0 / np.maximum(np.asarray(white, dtype=np.float32), min_white)
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)

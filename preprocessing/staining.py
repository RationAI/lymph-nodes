"""Cytokeratin (DAB) staining by colour deconvolution.

One definition of a stained cell for the refined annotation masks
(preprocessing/refined_annotation_masks.py) and the tiles' brown coverage
(preprocessing.tiling_blocks.DabCoverage), so the coverage counts the same cells the
refined labels are made of; and the stain densities of the lymph node masks
(preprocessing/node_masks.py).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


# Ruifrok & Johnston's haematoxylin / eosin / DAB stain vectors, as in
# skimage.color.rgb2hed; the DAB row of the inverse projects optical density onto DAB.
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED).astype(np.float32)
_DAB_FROM_RGB = _HED_FROM_RGB[:, 2]


def hed_densities(rgb: np.ndarray) -> np.ndarray:
    """Haematoxylin, eosin and DAB density of each pixel of ``rgb`` (..., 3): rgb2hed's."""
    optical = np.log(np.maximum(np.asarray(rgb, np.float32) / 255, 1e-6)) / np.log(1e-6)
    return np.maximum(optical @ _HED_FROM_RGB, 0)


def dab_density(rgb: np.ndarray) -> np.ndarray:
    """DAB density of each pixel of ``rgb`` (..., 3, uint8): skimage's rgb2hed DAB channel.

    Unlike a fixed RGB rule for "brown", it counts faint DAB and the near-black rims of
    strongly stained cells, and no haematoxylin.
    """
    optical = np.log(np.maximum(rgb.astype(np.float32) / 255, 1e-6)) / np.log(1e-6)
    return np.maximum(optical @ _DAB_FROM_RGB, 0)


def stained_cells(
    rgb: np.ndarray,
    mpp: float,
    threshold: float = 0.03,
    low_threshold: float = 0.022,
    min_area_um2: float = 10.0,
    hole_area_um2: float = 200.0,
    grow_um: float = 0.0,
) -> np.ndarray:
    """The cytokeratin-stained cells of ``rgb`` (H, W, 3) at ``mpp`` (cells_from_density)."""
    return cells_from_density(
        dab_density(rgb),
        mpp,
        threshold=threshold,
        low_threshold=low_threshold,
        min_area_um2=min_area_um2,
        hole_area_um2=hole_area_um2,
        grow_um=grow_um,
    )


def cells_from_density(
    density: np.ndarray,
    mpp: float,
    threshold: float = 0.03,
    low_threshold: float = 0.022,
    min_area_um2: float = 10.0,
    hole_area_um2: float = 200.0,
    grow_um: float = 0.0,
) -> np.ndarray:
    """The stained cells of a DAB ``density`` image (H, W) at ``mpp``, as a boolean mask.

    Pixels with DAB density above ``threshold``, extended by hysteresis to the
    connected pixels above ``low_threshold`` (a cell's faint rim and cytoplasm; the
    haematoxylin of dense lymphoid tissue reads ~0.01-0.02, so lower floods it),
    without specks under ``min_area_um2``; holes up to ``hole_area_um2`` filled —
    cytokeratin stains membrane and cytoplasm, so a cell's nucleus is an unstained hole
    in it, while a gland's lumen is larger and stays open — then grown by ``grow_um``
    (a disk). Holes touching the border of the image are left as they are: their
    extent is unknown.
    """
    labels, _ = ndimage.label(density > low_threshold)
    seeds = np.unique(labels[density > threshold])
    stained = ndimage.binary_opening(np.isin(labels, seeds[seeds > 0]))
    labels, _ = ndimage.label(stained)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area_um2 / mpp**2
    keep[0] = False
    stained = keep[labels]

    holes, _ = ndimage.label(~stained)
    sizes = np.bincount(holes.ravel())
    small = sizes <= hole_area_um2 / mpp**2
    border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    small[border] = False
    small[0] = False
    stained |= small[holes]

    if grow_um > 0 and stained.any():
        stained = ndimage.distance_transform_edt(~stained) <= grow_um / mpp
    return stained

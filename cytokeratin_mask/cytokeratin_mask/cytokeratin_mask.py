import pyvips
from skimage.filters import threshold_isodata, threshold_minimum
from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_holes,
    remove_small_objects,
)

from cytokeratin_mask.stain_processing import compute_stain_matrix, isolate_stain


def cytokeratin_mask(
    image: pyvips.Image, mask_min_area: int, holes_min_area: int
) -> pyvips.Image:
    ce_matrices = compute_stain_matrix(image, stains=["hematoxylin", "dab", "null"])
    image = isolate_stain(image, ce_matrices, 1)

    try:
        thresh = threshold_minimum(image)
    except RuntimeError:
        thresh = threshold_isodata(image)

    mask = image > thresh

    mask = binary_closing(mask, disk(3))
    remove_small_objects(mask, mask_min_area, out=mask)
    remove_small_holes(mask, holes_min_area, out=mask)
    return mask

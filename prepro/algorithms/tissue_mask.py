import pyvips


def tissue_mask(slide: pyvips.Image, disk_size: int = 10) -> pyvips.Image:
    """Generates a tissue mask from a whole-slide image (WSI) using saturation channel extraction and morphological operations, and saves the mask as a TIFF image.

    The function extracts the saturation channel from the WSI, applies Otsu thresholding to
    identify tissue regions, and performs morphological operations (closing and opening)
    to refine the mask.

    Args:
        slide: whole-slide image (WSI) pyvips file handler.
        disk_size: Size of the disk element for morphological operations (default is 10).

    Returns:
        The generated tissue mask as pyvips.Image.
    """
    # Extract saturation channel
    vi_slide_hsv = slide.sRGB2HSV()  # type: ignore pyvips
    _, vi_s, vi_v, *_ = vi_slide_hsv.bandsplit()  # type: ignore pyvips

    tresholded = (vi_v > 50) & ((vi_v < 240) | (vi_s > 10))

    # Morph Object
    vi_disk_object = pyvips.Image.black(2 * disk_size + 1, 2 * disk_size + 1) + 128  # type: ignore pyvips
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

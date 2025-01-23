from pathlib import Path

import pyvips


def color_separation(slide_path: Path) -> pyvips.Image:
    slide = pyvips.Image.new_from_file(slide_path, page=0, access="sequential")
    hue, saturation, value = slide.sRGB2HSV().bansplit()

    # Define the HSV range for brownish spots
    lower_hue, upper_hue = 10, 30
    lower_sat, upper_sat = 50, 255
    lower_val, upper_val = 50, 200

    tresholded = (
        (hue > lower_hue)
        & (hue < upper_hue)
        & (saturation > lower_sat)
        & (saturation < upper_sat)
        & (value > lower_val)
        & (value < upper_val)
    )

    # # Morph Object
    # vi_disk_object = pyvips.Image.black(2 * disk_size + 1, 2 * disk_size + 1) + 128  # type: ignore pyvips
    # vi_disk_object = vi_disk_object.draw_circle(
    #     255, disk_size, disk_size, disk_size, fill=True
    # )

    # # Closing
    # vi_mask = tresholded.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)
    # vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)

    # # Opening
    # vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.ERODE)
    # vi_mask = vi_mask.morph(vi_disk_object, pyvips.enums.OperationMorphology.DILATE)

    return tresholded

from typing import Any

import cv2
import numpy as np
import pyvips


def tissue_slicer(mask: pyvips.Image) -> Any:
    img = np.ndarray(
        buffer=mask.write_to_memory(), dtype=np.uint8, shape=(mask.height, mask.width)
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )

    return stats

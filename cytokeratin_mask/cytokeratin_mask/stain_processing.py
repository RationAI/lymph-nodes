import numpy as np
from numpy.typing import NDArray
from skimage import img_as_float, img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.util import invert

from cytokeratin_mask.histomicstk.color_deconvolution import color_deconvolution
from cytokeratin_mask.histomicstk.complement_stain_matrix import complement_stain_matrix
from cytokeratin_mask.histomicstk.separate_stains_xu_snmf import separate_stains_xu_snmf
from cytokeratin_mask.histomicstk.stain_color_map import stain_color_map
from cytokeratin_mask.histomicstk.utils import rgb_to_sda


def compute_stain_matrix(image: NDArray, stains: list[str]) -> NDArray:
    """Compute stain matrix adaptively.

    Returns:
        NDArray: stain matrix
    """
    im_input = img_as_ubyte(image)
    stain_matrix = np.array([stain_color_map[st] for st in stains]).T[:, :2]

    # Compute stain matrix adaptively
    sparsity_factor = 0.5
    i_0 = 230

    im_sda = rgb_to_sda(im_input, i_0)
    stain_matrix = separate_stains_xu_snmf(
        im_sda,
        stain_matrix,
        sparsity_factor,
    )
    return stain_matrix


def isolate_stain(image: NDArray, matrix: NDArray, i: int, i_o: int = 230) -> NDArray:
    """Isolate stain as grayscale image.

    Args:
        image (NDArray): Image to be processed.
        matrix (NDArray): Stain matrix.
        i (int): Index of stain to be isolated in stain matrix
        i_o (int, optional): Background RGB intensities. Defaults to 230.
    """
    im_deconvolved = color_deconvolution(
        img_as_ubyte(image),
        complement_stain_matrix(matrix),
        i_o,
    )
    r = im_deconvolved.Stains[:, :, i]
    res = rescale_intensity(img_as_float(r), out_range=(0, 1))

    return invert(res)

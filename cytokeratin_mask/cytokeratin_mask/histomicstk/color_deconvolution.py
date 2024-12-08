import collections

import numpy as np

from cytokeratin_mask.histomicstk.complement_stain_matrix import complement_stain_matrix
from cytokeratin_mask.histomicstk.utils import (
    convert_image_to_matrix,
    convert_matrix_to_image,
    normalize,
    rgb_to_sda,
    sda_to_rgb,
)


def color_deconvolution(im_rgb, w, i_0=None):
    """Perform color deconvolution.

    The given RGB Image `I` is first first transformed into optical density
    space, and then projected onto the stain vectors in the columns of the
    3x3 stain matrix `W`.

    For deconvolving H&E stained image use:

    `w` = array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    Parameters
    ----------
    im_rgb : array_like
        Input RGB Image that needs to be deconvolved.
    w : array_like
        A 3x3 matrix containing the color vectors in columns.
        For two stain images the third column is zero and will be
        complemented using cross-product. At least two of the three
        columns must be non-zero.
    i_0 : float or array_like, optional
        A float a 3-vector containing background RGB intensities.
        If unspecified, use the old OD conversion.

    Returns:
    -------
    Stains : array_like
        An rgb image where in each channel contains the image of the
        stain of the corresponding column in the stain matrix `W`.
        The intensity range of each channel is [0, 255] suitable for
        displaying.
    StainsFloat : array_like
        An intensity image of deconvolved stains that is unbounded,
        suitable for reconstructing color images of deconvolved stains
        with color_convolution.
    Wc : array_like
        A 3x3 complemented stain matrix. Useful for color image
        reconstruction with color_convolution.

    See Also:
    --------
    histomicstk.preprocessing.color_deconvolution.complement_stain_matrix,
    histomicstk.preprocessing.color_deconvolution.color_convolution
    histomicstk.preprocessing.color_conversion.rgb_to_od
    histomicstk.preprocessing.color_conversion.od_to_rgb
    histomicstk.preprocessing.color_conversion.rgb_to_sda
    histomicstk.preprocessing.color_conversion.sda_to_rgb

    """
    # complement stain matrix if needed
    w = np.array(w)
    if w.shape[1] < 3:
        wc = np.zeros((w.shape[0], 3))
        wc[:, : w.shape[1]] = w
        w = wc

    wc = complement_stain_matrix(w) if np.linalg.norm(w[:, 2]) <= 1e-16 else w

    # normalize stains to unit-norm
    wc = normalize(wc)

    # invert stain matrix
    q = np.linalg.pinv(wc)

    # transform 3D input image to 2D RGB matrix format
    m = convert_image_to_matrix(im_rgb)[:3]

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = rgb_to_sda(m, i_0)
    sda_deconv = np.dot(q, sda_fwd)
    sda_inv = sda_to_rgb(sda_deconv, 255 if i_0 is not None else None)

    # reshape output
    StainsFloat = convert_matrix_to_image(sda_inv, im_rgb.shape)

    # transform type
    Stains = StainsFloat.clip(0, 255).astype(np.uint8)

    # return
    Unmixed = collections.namedtuple("Unmixed", ["Stains", "StainsFloat", "Wc"])
    Output = Unmixed(Stains, StainsFloat, wc)

    return Output

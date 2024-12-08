import numpy as np


def exclude_nonfinite(m):
    """Exclude columns from m that have infinities or nans.  In the
    context of color deconvolution, these occur in conversion from RGB
    to SDA when the source has 0 in a channel.
    """
    return m[:, np.isfinite(m).all(axis=0)]


def convert_image_to_matrix(im):
    """Convert an image (MxNx3 array) to a column matrix of pixels
    (3x(M*N)).  It will pass through a 2D array unchanged.

    """
    if im.ndim == 2:
        return im

    return im.reshape((-1, im.shape[-1])).T


def convert_matrix_to_image(m, shape):
    """Convert a column matrix of pixels to a 3D image given by shape.
    The number of channels is taken from m, not shape.  If shape has
    length 2, the matrix is returned unchanged.  This is the inverse
    of convert_image_to_matrix:

    im == convert_matrix_to_image(convert_image_to_matrix(im),
    im.shape)

    """
    if len(shape) == 2:
        return m

    return m.T.reshape(shape[:-1] + (m.shape[0],))


def magnitude(m):
    """Get the magnitude of each column vector in a matrix"""
    return np.sqrt((m**2).sum(0))


def normalize(m):
    """Normalize each column vector in a matrix"""
    return m / magnitude(m)


def rgb_to_sda(im_rgb, I_0, allow_negatives=False):
    """Transform input RGB image or matrix `im_rgb` into SDA (stain
    darkness) space for color deconvolution.

    Parameters
    ----------
    im_rgb : array_like
        Image (MxNx3) or matrix (3xN) of pixels

    I_0 : float or array_like
        Background intensity, either per-channel or for all channels

    allow_negatives : bool
        If False, would-be negative values in the output are clipped to 0

    Returns:
    -------
    im_sda : array_like
        Shaped like `im_rgb`, with output values 0..255 where `im_rgb` >= 1

    Note:
    ----
    For compatibility purposes, passing I_0=None invokes the behavior of
    rgb_to_od.

    See Also:
    --------
    histomicstk.preprocessing.color_conversion.sda_to_rgb,
    histomicstk.preprocessing.color_conversion.rgb_to_od,
    histomicstk.preprocessing.color_deconvolution.color_deconvolution,
    histomicstk.preprocessing.color_deconvolution.color_convolution

    """
    is_matrix = im_rgb.ndim == 2
    if is_matrix:
        im_rgb = im_rgb.T

    if I_0 is None:  # rgb_to_od compatibility
        im_rgb = im_rgb.astype(float) + 1
        I_0 = 256

    im_rgb = np.maximum(im_rgb, 1e-10)

    im_sda = -np.log(im_rgb / (1.0 * I_0)) * 255 / np.log(I_0)
    if not allow_negatives:
        im_sda = np.maximum(im_sda, 0)
    return im_sda.T if is_matrix else im_sda


def sda_to_rgb(im_sda, I_0):
    """Transform input SDA image or matrix `im_sda` into RGB space.  This
    is the inverse of `rgb_to_sda` with respect to the first parameter

    Parameters
    ----------
    im_sda : array_like
        Image (MxNx3) or matrix (3xN) of pixels

    I_0 : float or array_like
        Background intensity, either per-channel or for all channels

    Note:
    ----
    For compatibility purposes, passing I_0=None invokes the behavior of
    od_to_rgb.

    See Also:
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_sda,
    histomicstk.preprocessing.color_conversion.od_to_rgb,
    histomicstk.preprocessing.color_deconvolution.color_deconvolution,
    histomicstk.preprocessing.color_deconvolution.color_convolution

    """
    is_matrix = im_sda.ndim == 2
    if is_matrix:
        im_sda = im_sda.T

    od = I_0 is None
    if od:  # od_to_rgb compatibility
        I_0 = 256

    im_rgb = I_0 ** (1 - im_sda / 255.0)
    return (im_rgb.T if is_matrix else im_rgb) - od

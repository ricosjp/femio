
import numpy as np

from . import config


def normalize(array, keep_zeros=False):
    """Normalizes input array.

    Parameters
    ----------
    array: numpy.ndarray
        2-dimensional input array
    keep_zeros: bool, optional
        If True, keep zero vectors as is. The default is False.

    Returns
    -------
    numpy.ndarray:
        Normalized array
    """
    if len(array.shape) != 2:
        raise ValueError(
            f"Input array should be 2-dim array ({array.shape} is given)")

    norms = np.linalg.norm(array, axis=1)[:, None]
    if keep_zeros:
        norms[norms < config.EPSILON] = 1.
    else:
        norms[norms < config.EPSILON] = config.EPSILON
    return array / norms

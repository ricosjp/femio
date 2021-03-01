
import numpy as np
import scipy.sparse as sp

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


def align_nnz(a, ref):
    """Align a sparse matrix a to have the same nnz profile as ref.

    Parameters
    ----------
    a: scipy.sparse.csr_matrix or scipy.sparse.coo_matrix
    ref: scipy.sparse.csr_matrix or scipy.sparse.coo_matrix

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    if isinstance(a, sp.csr_matrix) and isinstance(ref, sp.csr_matrix):
        dummy_array = np.ones(len(ref.data))
        dummy_csr = sp.csr_matrix(
            (dummy_array, ref.indices, ref.indptr), shape=ref.shape)
        added_a = a + dummy_csr
        reduced_array = added_a.data - dummy_array
        aligned_a = sp.csr_matrix(
            (reduced_array, ref.indices, ref.indptr), shape=ref.shape)
        return aligned_a

    if isinstance(a, sp.coo_matrix) and isinstance(ref, sp.coo_matrix):
        return align_nnz(a.tocsr(), ref.tocsr())
    else:
        raise NotImplementedError(
            f"Type: {a.__class__}, {ref.__class__}")

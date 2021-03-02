
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


def align_nnz(sparses):
    """Align a sparse matrices to have the same nnz profile as ref.

    Parameters
    ----------
    sparses: List[scipy.sparse.csr_matrix] or List[scipy.sparse.coo_matrix]

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    shapes = np.array([s.shape for s in sparses])
    if not np.all(shapes == shapes[0]):
        raise ValueError(f"Inputs should have the same shape: {shapes}")

    if np.all([isinstance(s, sp.csr_matrix) for s in sparses]):
        # Determine dummy_scale to avoid zero after summation
        dummy_scale = np.abs(np.min([np.min(s) for s in sparses])) * 2 + 1
        dummy_csr = sp.csr_matrix(sparses[0].shape)
        for s in sparses:
            dummy_csr = dummy_csr + sp.csr_matrix(
                (np.ones(len(s.data)) * dummy_scale, s.indices, s.indptr),
                shape=s.shape)
        dummy_csr.sort_indices()
        dummy_array = dummy_csr.data
        added_sparses = [s + dummy_csr for s in sparses]
        for added_sparse in added_sparses:
            added_sparse.sort_indices()
        reduced_arrays = [a_s.data - dummy_array for a_s in added_sparses]
        aligned_sparses = [
            sp.csr_matrix(
                (reduced_array, dummy_csr.indices, dummy_csr.indptr),
                shape=dummy_csr.shape) for reduced_array in reduced_arrays]
        return aligned_sparses

    else:
        return align_nnz([s.tocsr() for s in sparses])

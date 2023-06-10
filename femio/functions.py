
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


def remove_duplicates(
        connectivities, return_index=False, return_inverse=False, end=None):
    """Remove duplicating elements.

    Parameters
    ----------
    connectivities: numpy.ndarray[int]
        Element connectivities.
    return_index: bool, optional
        If True, return also indices of unique.
    return_inverse: bool, optional
        If True, return also the inverse_indice
    end: int, optional
        If fed, use only first `end` elements to detect duplication.

    Returns
    -------
    connectivities_wo_duplications: numpy.ndarray[int]
        Element connectivities without duplications.
    """
    sorted_connectivities = [
        np.sort(connectivity)[:end] for connectivity in connectivities]
    unique = np.unique(
        sorted_connectivities, axis=0,
        return_index=True, return_inverse=return_inverse)
    indices = unique[1]
    ret = [connectivities[indices]]

    if return_index:
        ret.append(indices)
    if return_inverse:
        ret.append(unique[-1])
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def convert_array2symmetric_matrix(
        in_array, *,
        from_engineering=False, order=None):
    """Convert (n, 6) shaped array to (n, 3, 3) symmetric matrix.

    Args:
        in_array: numpy.ndarray
            (n, 6) shaped array.
        from_engineering: bool, optional [False]
            If True, treat in_array as engineering-strain-like data.
        order: array-like, optional [[0, 1, 2, 3, 4, 5]]
            The order of in_array. The order should be
            [11, 22, 33, 12, 23, 31] order.
    Returns:
        symmetrc_matrix: numpy.ndarray
            (n, 6, 6) symmetric matrix.
    """
    if order is None:
        order = [0, 1, 2, 3, 4, 5]
    in_array = in_array[:, order]
    if from_engineering:
        # Handle engineering strain stuff
        in_array[:, 3:] = in_array[:, 3:] / 2

    indices_array2symmetric_matrix = [0, 3, 5, 3, 1, 4, 5, 4, 2]
    return np.reshape(
        in_array[:, indices_array2symmetric_matrix], (-1, 3, 3))


def convert_symmetric_matrix2array(
        in_matrix, *,
        to_engineering=False, order=None):
    """Convert (n, 3, 3) shaped symmetric matrices to (n, 6) array.

    Args:
        in_matrix: numpy.ndarray
            (n, 6, 6) symmetric matrix.
        to_engineering: bool, optional [False]
            If True, the out_array will be converted to
            engineering-strain-like data.
        order: array-like, optional [[0, 1, 2, 3, 4, 5]]
            The order of out_array. The order should be
            [11, 22, 33, 12, 23, 31] order.
    Returns:
        out_array: numpy.ndarray
            (n, 6) shaped array.
    """
    if order is None:
        order = [0, 1, 2, 3, 4, 5]

    indices_symmetric_matrix2array = [0, 4, 8, 1, 5, 2]
    out_array = np.reshape(in_matrix, (-1, 9))[
        :, indices_symmetric_matrix2array]

    if to_engineering:
        # Handle engineering strain stuff
        out_array[:, 3:] = out_array[:, 3:] * 2
    return out_array[:, order]


def calculate_symmetric_matrices_from_eigens(
        eigenvalues, eigenvectors):
    diags = convert_array2symmetric_matrix(np.concatenate(
        [eigenvalues, np.zeros(eigenvalues.shape)], axis=1))
    rotation_matrices = np.stack([
        eigenvectors[:, :3],
        eigenvectors[:, 3:6],
        eigenvectors[:, 6:],
    ], axis=2)
    return rotation_matrices @ diags @ np.transpose(
        rotation_matrices, (0, 2, 1))


def invert_strain(strain, is_engineering=False):
    values, directions, vectors = calculate_principal_components(
        strain, from_engineering=is_engineering)
    # inverted_values = np.exp(- np.log(1 + values)) - 1
    inverted_values = 1. / (1. + values) - 1.
    return calculate_array_from_eigens(
        inverted_values, directions, to_engineering=is_engineering)


def calculate_array_from_eigens(
        eigenvalues, eigenvectors, to_engineering=False):
    return convert_symmetric_matrix2array(
        calculate_symmetric_matrices_from_eigens(
            eigenvalues, eigenvectors), to_engineering=to_engineering)


def calculate_principal_components(
        in_array, *,
        from_engineering=False, order=None):
    """Calculate eigenvalues and eigenvectors of the input arrays which are
    parts of symmetric matrices.

    Args:
        in_array: numpy.ndarray
            (n, 6) shaped array to form symmetric matrices.
        from_engineering: bool, optional [False]
            If True, treat in_array as engineering-strain-like data.
        order: array-like, optional [[0, 1, 2, 3, 4, 5]]
            The order of in_array. The order should be
            [11, 22, 33, 12, 23, 31] order.
    Returns:
        eigenvalues: numpy.ndarray
            (n, 3) shaped array of eigenvalues.
        eigenvectors: numpy.ndarray
            (n, 9) shaped array of eigenvectors.
    """
    symmetrc_matrix = convert_array2symmetric_matrix(
        in_array, from_engineering=from_engineering, order=order)
    _eigenvalues, _directions = np.linalg.eigh(symmetrc_matrix)

    # Sort descending order
    eigenvalues = _eigenvalues[:, ::-1]
    directions = _directions[:, :, ::-1]

    # Make sure it right handed system
    directions[:, :, 2] = np.cross(
        directions[:, :, 0], directions[:, :, 1])

    vectors = np.einsum(
        'ik,ijk->ijk', eigenvalues, directions)
    return (
        eigenvalues,
        np.concatenate([
            directions[:, :, 0], directions[:, :, 1],
            directions[:, :, 2]], axis=1),
        np.concatenate([
            vectors[:, :, 0], vectors[:, :, 1],
            vectors[:, :, 2]], axis=1))

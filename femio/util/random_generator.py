import itertools

import numpy as np
from scipy.spatial import Delaunay

from .. import fem_attribute
from .. import fem_elemental_attribute
from .. import fem_data


def generate_random_mesh(
        element_type, n_point,
        *, x_length=1., y_length=1., z_length=1., quality_threshold=None,
        noise_scale=1., strip_epsilon=None):
    """Generate random mesh.

    Parameters
    ----------
    element_type: str
        Element type selected from ['tri', 'quad', 'tet', 'hex'].
    n_point: int
        The number of points.
    x_length: float, optional
        The maximum X length of the mesh. The default is 1.
    y_length: float, optional
        The maximum Y length of the mesh. The default is 1.
    z_length: float, optional
        The maximum Z length of the mesh. The default is 1.
    quality_threshold: float, optional
        If fed, compute quality of elements and remove some when the quality
        is bad.
    noise_scale: float, optional
        The scale of the noise added to node positions. The default is 0.1.
    strip_epsilon: float, optional
        If True, remove superfacial elements, which tend to have bad quality.

    Returns
    -------
    fem_data: femio.FEMData
        FEMData object of the generated brick mesh.
    """
    if element_type == 'tri':
        dim = 2
        scales = [x_length, y_length]
    elif element_type == 'tet':
        dim = 3
        scales = [x_length, y_length, z_length]
    else:
        raise ValueError(f"Unexpected element type: {element_type}")

    node_positions, element_connectivities = _generate_random(
        dim=dim, n_point=n_point, scales=scales,
        noise_scale=noise_scale)

    if quality_threshold is not None:
        score = _compute_scores(node_positions, element_connectivities - 1)
        quality_filter = score > quality_threshold
    else:
        quality_filter = np.ones(len(element_connectivities), dtype=bool)

    if strip_epsilon is not None:
        strip_filter = _compute_strip_filter(
            node_positions, element_connectivities - 1, scales, strip_epsilon)
    else:
        strip_filter = np.ones(len(element_connectivities), dtype=bool)

    filter_ = np.logical_and(quality_filter, strip_filter)
    element_connectivities = element_connectivities[filter_]
    # filter_ = np.ones(len(element_connectivities), dtype=bool)
    nodes = fem_attribute.FEMAttribute(
        'NODE', ids=np.arange(len(node_positions))+1, data=node_positions)
    elements = fem_elemental_attribute.FEMElementalAttribute(
        'ELEMENT', data=element_connectivities, element_type=element_type)
    fd = fem_data.FEMData(nodes=nodes, elements=elements)

    if quality_threshold is not None:
        fd.elemental_data.update_data(
            fd.elements.ids, {'score': score[filter_, None]})
    if strip_filter is not None:
        fd.elemental_data.update_data(
            fd.elements.ids,
            {'strip': strip_filter[filter_, None].astype(int)})

    fd.make_elements_positive()
    fd.remove_useless_nodes()
    fd.elemental_data.pop('metric')
    fd.elemental_data.pop('volume')
    return fd


def _generate_random(dim, n_point, scales, noise_scale):
    density = n_point**(1/dim) / np.prod(scales)**(1/dim)
    if dim == 2:
        x, y = np.meshgrid(
            np.linspace(0., scales[0], max(int(density * scales[0]), 2)),
            np.linspace(0., scales[1], max(int(density * scales[1]), 2)))
        node_positions = np.stack([
            np.ravel(x), np.ravel(y), np.zeros(np.ravel(x).shape)], axis=-1)
    elif dim == 3:
        # Permute arguments to have understantable order
        y, z, x = np.meshgrid(
            np.linspace(0., scales[1], max(int(density * scales[1]), 2)),
            np.linspace(0., scales[2], max(int(density * scales[2]), 2)),
            np.linspace(0., scales[0], max(int(density * scales[0]), 2)))
        node_positions = np.stack([
            np.ravel(x), np.ravel(y), np.ravel(z)], axis=-1)
    else:
        raise ValueError(f"Unexpected dim: {dim}")

    for i_coord in range(dim):
        node_positions[:, i_coord] = node_positions[:, i_coord] \
            + np.random.randn(len(node_positions)) \
            * scales[i_coord] * noise_scale / n_point**(1/dim)
    delaunay = Delaunay(node_positions[:, :dim])
    element_connectivities = delaunay.simplices + 1
    return node_positions, element_connectivities


def _compute_scores(node_positions, simplex_indices):
    dim = simplex_indices.shape[-1]
    list_simplex_nodes = [
        node_positions[simplex_indices[:, i]] for i in range(dim)]
    scores = np.min(np.arccos(np.abs(np.stack(
        [
            _compute_cos(n1 - n3, n2 - n3)
            for n1, n2, n3 in itertools.combinations(list_simplex_nodes, r=3)],
        axis=-1))), axis=-1) / np.pi * 2
    return scores


def _compute_cos(n1, n2):
    norm1 = np.linalg.norm(n1, axis=-1)
    norm2 = np.linalg.norm(n2, axis=-1)
    dot = np.einsum('ij,ij->i', n1, n2)
    return dot / norm1 / norm2


def _compute_strip_filter(node_positions, simplex_indices, scales, epsilon):
    dim = node_positions.shape[-1]
    list_node_components = [
        node_positions[simplex_indices, i] for i in range(dim)]
    return np.prod(np.stack(
        [
            np.prod(
                np.logical_and(
                    epsilon * scales[i_component] < node_component,
                    node_component < scales[i_component] * (1 - epsilon)),
                axis=-1).astype(bool)
            for i_component, node_component
            in enumerate(list_node_components)],
        axis=-1), axis=-1).astype(bool)

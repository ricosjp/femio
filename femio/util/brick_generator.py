
import numpy as np

from .. import fem_attribute
from .. import fem_elemental_attribute
from .. import fem_data


def generate_brick(
        element_type, n_x_element, n_y_element, n_z_element=None,
        *, x_length=1., y_length=1., z_length=1.):
    """Generate brick mesh.

    Parameters
    ----------
    element_type: str
        Element type selected from ['tri', 'quad', 'tet', 'hex'].
    n_x_element: int
        The number of element in the X direction.
    n_y_element: int
        The number of element in the Y direction.
    n_z_element: int, optional
        The number of element in the Z direction.
    x_length: float, optional
        The X length of the brick. The default is 1.
    y_length: float, optional
        The Y length of the brick. The default is 1.
    z_length: float, optional
        The Z length of the brick. The default is 1.

    Returns
    -------
    fem_data: femio.FEMData
        FEMData object of the generated brick mesh.
    """
    if element_type in ['tri', 'quad']:
        node_positions, element_connectivities = _generate_brick_2d(
            element_type, n_x_element, n_y_element,
            x_length=x_length, y_length=y_length)
    elif element_type in ['tet', 'hex']:
        node_positions, element_connectivities = _generate_brick_3d(
            element_type, n_x_element, n_y_element, n_z_element,
            x_length=x_length, y_length=y_length, z_length=z_length)
    else:
        raise ValueError(f"Unexpected element type: {element_type}")

    nodes = fem_attribute.FEMAttribute(
        'NODE', ids=np.arange(len(node_positions))+1, data=node_positions)
    elements = fem_elemental_attribute.FEMElementalAttribute(
        'ELEMENT', data=element_connectivities, element_type=element_type)
    return fem_data.FEMData(nodes=nodes, elements=elements)


def _generate_brick_2d(
        element_type, n_x_element, n_y_element, x_length=1., y_length=1.):
    x, y = np.meshgrid(
        np.linspace(0., x_length, n_x_element + 1),
        np.linspace(0., y_length, n_y_element + 1))
    node_positions = np.stack([
        np.ravel(x), np.ravel(y), np.zeros(np.ravel(x).shape)], axis=-1)

    # Generate elements
    n_x = n_x_element + 1
    n_y = n_y_element + 1

    if element_type == 'tri':
        def generate_element(i):
            return np.array([
                [i, i + 1, i + 1 + n_x],
                [i, i + 1 + n_x, i + n_x]
            ])
    elif element_type == 'quad':
        def generate_element(i):
            return np.array([
                [i, i + 1, i + 1 + n_x, i + n_x],
            ])
    else:
        raise ValueError(f"Unexpected element_type: {element_type}")

    element_connectivities = np.concatenate([
        generate_element(i)
        for i in range(n_x * n_y)
        if (i+1) % n_x != 0 and i < n_x * n_y_element]) + 1
    return node_positions, element_connectivities


def _generate_brick_3d(
        element_type, n_x_element, n_y_element, n_z_element,
        x_length=1., y_length=1., z_length=1.):
    # Permute arguments to have understantable order
    y, z, x = np.meshgrid(
        np.linspace(0., y_length, n_y_element + 1),
        np.linspace(0., z_length, n_z_element + 1),
        np.linspace(0., x_length, n_x_element + 1))
    node_positions = np.stack([
        np.ravel(x), np.ravel(y), np.ravel(z)], axis=-1)

    # Generate elements
    n_x = n_x_element + 1
    n_y = n_y_element + 1
    n_z = n_z_element + 1
    n_xy = n_x * n_y

    if element_type == 'tet':
        def generate_element(i):
            i1 = i
            i2 = i + 1
            i3 = i + n_x + 1
            i4 = i + n_x
            i5 = i + n_xy
            i6 = i + n_xy + 1
            i7 = i + n_xy + n_x + 1
            i8 = i + n_xy + n_x
            return np.array([
                [i1, i2, i4, i5],
                [i6, i5, i7, i2],
                [i2, i5, i7, i4],
                [i2, i3, i4, i7],
                [i5, i8, i7, i4],
            ])
    elif element_type == 'hex':
        def generate_element(i):
            return np.array([
                [
                    i, i + 1, i + 1 + n_x, i + n_x,
                    i + n_xy, i + n_xy + 1, i + n_xy + 1 + n_x, i + n_xy + n_x,
                ],
            ])
    else:
        raise ValueError(f"Unexpected element_type: {element_type}")

    element_connectivities = np.concatenate([
        generate_element(i)
        for i in range(n_x * n_y * n_z)
        if (i+1) % n_x != 0 and (i+1) % n_xy < 1 + n_xy - n_x
        and i < n_xy * n_z_element]) + 1
    return node_positions, element_connectivities

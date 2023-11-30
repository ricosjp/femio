
import numpy as np

from . import config
from .fem_attribute import FEMAttribute
from .fem_elemental_attribute import FEMElementalAttribute
from .fem_data import FEMData


def read_directory(*args, **kwargs):
    """Initialize FEMData object from directory.

    See FEMData.read_directory for more detail.
    """
    return FEMData.read_directory(*args, **kwargs)


def read_npy_directory(*args, **kwargs):
    """Initialize FEMData object from directory that contains femio npy data.

    See FEMData.read_npy_directory for more detail.
    """
    return FEMData.read_npy_directory(*args, **kwargs)


def read_files(*args, **kwargs):
    """Initialize FEMData object from files.

    See FEMData.read_files for more detail.
    """
    return FEMData.read_files(*args, **kwargs)


def convert_vtk_unstructured_to_femio(mesh):
    """Convert vtk unstructured grid to femio.FEMData object.

    Parameters
    ----------
    mesh: tvtk.tvtk_classes.unstructured_grid.UnstructuredGrid
        Input UnstructuredGrid object.

    Returns
    -------
    femio.FEMData
    """
    # Read nodes
    node_positions = mesh.points.to_array()
    nodes = FEMAttribute(
        'NODE', ids=np.arange(len(node_positions))+1, data=node_positions)

    # Read elements
    cells = mesh.get_cells()
    cell_types = mesh.cell_types_array.to_array().astype(np.int32)
    connectivity = cells.connectivity_array.to_array().astype(np.int32)
    offsets = cells.offsets_array.to_array().astype(np.int32)
    element_connectivities = np.array(
        [connectivity[o1:o2] for o1, o2 in zip(offsets[:-1], offsets[1:])],
        dtype=object)
    global_element_ids = np.arange(len(cell_types)) + 1
    unique_cell_types = np.unique(cell_types)
    elements_data = {
        config.DICT_VTK_ID_TO_ELEMENT_TYPE[t]: FEMAttribute(
            config.DICT_VTK_ID_TO_ELEMENT_TYPE[t],
            ids=global_element_ids[cell_types == t],
            data=stack_if_needed(
                t, element_connectivities[cell_types == t]) + 1)
        for t in unique_cell_types}
    update_voxel2hex_if_needed(elements_data)
    elements = FEMElementalAttribute('ELEMENT', elements_data)

    obj = FEMData(nodes=nodes, elements=elements)
    raw_faces = mesh.get_faces()
    if raw_faces is not None:
        faces = mesh.get_faces().to_array().astype(np.int32)
        face_offsets = np.concatenate([
            mesh.face_locations.to_array(),
            np.array([len(faces)])]).astype(np.int32)
        face_offsets = face_offsets[face_offsets != -1]
        face_data = np.array([0] + [
            list(faces[l1:l2]) for l1, l2
            in zip(face_offsets[:-1], face_offsets[1:])], dtype=object)[1:]
        obj.elemental_data.update({
            'face': FEMElementalAttribute(
                'face', {
                    'polyhedron':
                    FEMAttribute(
                        'face', ids=obj.elements.ids[cell_types == 42],
                        data=face_data)})})

    # Read point data
    obj.nodal_data.update_data(
        obj.nodes.ids, {
            mesh.point_data.get_array_name(i_array):
            convert_to_2d_if_needed(mesh.point_data.get_array(
                mesh.point_data.get_array_name(i_array)
            ).to_array())
            for i_array in range(
                mesh.point_data.trait_get(
                    'number_of_arrays')['number_of_arrays'])},
        allow_overwrite=True)

    # Read cell data
    obj.elemental_data.update_data(
        obj.elements.ids, {
            mesh.cell_data.get_array_name(i_array):
            convert_to_2d_if_needed(mesh.cell_data.get_array(
                mesh.cell_data.get_array_name(i_array)
            ).to_array())
            for i_array in range(
                mesh.cell_data.trait_get(
                    'number_of_arrays')['number_of_arrays'])},
        allow_overwrite=True)
    return obj


def update_voxel2hex_if_needed(elements_data):
    if 'voxel' not in elements_data:
        return elements_data
    voxel = elements_data.pop('voxel')
    d = voxel.data

    ids = voxel.ids
    data = np.stack([
        d[:, 0],
        d[:, 1],
        d[:, 3],
        d[:, 2],
        d[:, 4],
        d[:, 5],
        d[:, 7],
        d[:, 6],
    ], axis=-1)
    if 'hex' in elements_data:
        ids = np.concatenate([elements_data['hex'].ids, ids])
        data = np.concatenate([elements_data['hex'].data, data])

    elements_data.update({'hex': FEMAttribute(name='hex', ids=ids, data=data)})
    return


def convert_to_2d_if_needed(x):
    if len(x.shape) == 1:
        return x[:, None]
    else:
        return x


def stack_if_needed(type_id, data):
    if type_id in [42, 7]:
        return data
    else:
        return np.stack(data)

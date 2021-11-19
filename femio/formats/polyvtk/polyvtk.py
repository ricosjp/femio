
import pathlib

import numpy as np
from tvtk.api import tvtk

from ...fem_attribute import FEMAttribute
from ...fem_elemental_attribute import FEMElementalAttribute
from ...fem_data import FEMData


class PolyVTKData(FEMData):
    """FEMData of VTK with polyhedron."""

    DICT_VTK_ID_TO_ELEMENT_TYPE = {
        10: 'tet',
        12: 'hex',
        13: 'prism',
        14: 'pyr',
        42: 'polyhedron',
    }

    @classmethod
    def read_files(cls, file_names, read_mesh_only=False, time_series=False):
        """Initialize PolyVTKData object.

        Parameters
        ----------
        file_names: list of str
            File names.
        read_mesh_only: bool, optional
            If true, read mesh (nodes and elements) and ignore
            material data, results and so on. The default is False.
        """
        if isinstance(file_names, str):
            file_name = file_names
        else:
            file_name = file_names[0]

        if not pathlib.Path(file_name).is_file():
            raise ValueError(f"{file_name} not found")

        reader = tvtk.XMLUnstructuredGridReader(file_name=file_name)
        reader.update()
        mesh = reader.get_output()

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
            cls.DICT_VTK_ID_TO_ELEMENT_TYPE[t]: FEMAttribute(
                cls.DICT_VTK_ID_TO_ELEMENT_TYPE[t],
                ids=global_element_ids[cell_types == t],
                data=stack_if_needed(
                    t, element_connectivities[cell_types == t]) + 1)
            for t in unique_cell_types}
        elements = FEMElementalAttribute('ELEMENT', elements_data)

        obj = cls(nodes=nodes, elements=elements)
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


def convert_to_2d_if_needed(x):
    if len(x.shape) == 1:
        return x[:, None]
    else:
        return x


def stack_if_needed(type_id, data):
    if type_id == 42:
        return data
    else:
        return np.stack(data)

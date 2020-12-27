import io

import numpy as np
import stl

from ...fem_attribute import FEMAttribute
from ...fem_elemental_attribute import FEMElementalAttribute
from ...fem_data import FEMData


HEADER_SIZE = 80  # The amount of bytes in the header field


class STLData(FEMData):
    """FEMEntity of VTK version."""

    @classmethod
    def read_files(cls, file_names, read_mesh_only=False, time_series=False):
        """Initialize STLEntity object.

        Args:
            file_names: list of str
                File names.
            read_mesh_only: bool, optional [False]
                If true, read mesh (nodes and elements) and ignore
                material data, results and so on.
        """
        obj = cls()
        obj.file_names = file_names

        if len(file_names) != 1:
            raise ValueError(
                f"{len(file_names)} files found. "
                'Specify file name by using read_files() instead of '
                'read_directory().')
        file_name = file_names[0]

        print('Parsing data')
        # Test if the input file is ascii or not
        with open(file_name, 'rb') as f:
            header = f.read(HEADER_SIZE).lower()
            is_ascii = header.startswith(b'solid')

        if is_ascii:
            with open(file_name, 'r') as read_file:
                # Remove 'color' statement lines because
                # numpy-stl nor meshio cannot handle it
                string_io = io.StringIO(''.join([
                    line for line in read_file.readlines()
                    if 'color' not in line]))
            stl_meshes = stl.mesh.Mesh.from_multi_file(None, fh=string_io)
        else:
            stl_meshes = stl.mesh.Mesh.from_multi_file(file_name)
        points = np.concatenate([
            np.reshape(stl_mesh.vectors, (-1, 3)) for stl_mesh in stl_meshes])

        # Remove verbose points
        print('Maing unique')
        unique_points, inverse_indices = np.unique(
            points, axis=0, return_inverse=True)
        node_ids = np.arange(len(unique_points)) + 1
        obj.nodes = FEMAttribute(
            'NODE', ids=node_ids, data=unique_points)

        element_data = np.reshape(node_ids[inverse_indices], (-1, 3))
        element_ids = np.arange(len(element_data)) + 1
        obj.elements = FEMElementalAttribute(
            'ELEMENT',
            {'tri': FEMAttribute(
                'tri', ids=element_ids, data=element_data)})
        return obj

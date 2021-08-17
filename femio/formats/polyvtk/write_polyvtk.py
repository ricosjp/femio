
import fileinput

import numpy as np
from tvtk.api import tvtk

from . import polyvtk


class PolyVTKWriter():

    DICT_ELEMENT_TYPE_TO_VTK_ID = {
        v: k for k, v
        in polyvtk.PolyVTKData.DICT_VTK_ID_TO_ELEMENT_TYPE.items()}

    def __init__(self, fem_data, *, include_displacement=False):
        """Initialize PolyVTKWriter object.

        Parameters
        ----------
        fem_data: FEMData
            FEMData object to write.
        include_displacement: bool, optional
            If True, create STL with node + displacement instead of node only.
            The default is False.
        """
        self.fem_data = fem_data
        return

    def write(self, file_name=None, *, overwrite=False):
        """Write FEM data in VTK unstructured grid (vtu) format.

        Parameters
        ----------
        file_name: str
            File name of the output file. If not fed,
            input_filename.out.stl will be the output file name.
        overwrite: bool, optional [False]
            If True, allow averwrite files.
        """
        unstructured_grid = tvtk.UnstructuredGrid(
            points=self.fem_data.nodes.data)
        if 'polyhedron' in self.fem_data.elements:
            unstructured_grid.set_cells(
                self.DICT_ELEMENT_TYPE_TO_VTK_ID['polyhedron'],
                list(self.fem_data.elemental_data['face'][
                    'polyhedron'].data))

        for element_type, element_data in self.fem_data.elements.items():
            if element_type == 'polyhedron':
                continue
            element_type_id = self.DICT_ELEMENT_TYPE_TO_VTK_ID[element_type]
            for element in element_data.data - 1:
                unstructured_grid.insert_next_cell(element_type_id, element)

        # Add cell data
        n_element = len(self.fem_data.elements)
        for k, v in self.fem_data.elemental_data.items():
            if len(v) != n_element:
                continue
            if k == 'face':
                continue
            array_id = unstructured_grid.cell_data.add_array(
                self._reorder_cell_data(v.data))
            unstructured_grid.cell_data.get_array(array_id).name = k

        # Add point data
        n_node = len(self.fem_data.nodes)
        for k, v in self.fem_data.nodal_data.items():
            if len(v) != n_node:
                continue
            array_id = unstructured_grid.point_data.add_array(v.data)
            unstructured_grid.point_data.get_array(array_id).name = k

        writer = tvtk.XMLUnstructuredGridWriter(file_name=str(file_name))
        writer.data_mode = 'ascii'
        writer.set_input_data(unstructured_grid)
        writer.write()

        # Replace int64 with int32 because tvtk has trouble to read it
        with fileinput.input(file_name, inplace=True) as file:
            for line in file:
                print(line.replace('Int64', 'Int32'), end='')
        return

    def _reorder_cell_data(self, data):
        return np.concatenate(
            [data[self.fem_data.elements.types == 'polyhedron']] + [
                data[self.fem_data.elements.types == t]
                for t, _ in self.fem_data.elements.items()
                if t != 'polyhedron'])


def single_type_ug():
    """Simple example showing how to create an unstructured grid
    consisting of cells of a single type.
    """
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # tets
        [1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1],
        [2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1],
    ], 'f')
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    tet_type = tvtk.Tetra().cell_type
    ug = tvtk.UnstructuredGrid(points=points)
    ug.set_cells(tet_type, tets)
    return ug


def mixed_type_ug():
    """A slightly more complex example of how to generate an
    unstructured grid with different cell types.  Returns a created
    unstructured grid.
    """
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # tetra
        [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
        [2, 0, 1], [3, 0, 1], [3, 1, 1], [2, 1, 1],  # Hex
    ], 'f')
    # shift the points so we can show both.
    points[:, 1] += 2.0
    # The cells
    cells = np.array([
        4, 0, 1, 2, 3,  # tetra
        8, 4, 5, 6, 7, 8, 9, 10, 11  # hex
    ])
    # The offsets for the cells, i.e. the indices where the cells
    # start.
    offset = np.array([0, 5])
    tetra_type = tvtk.Tetra().cell_type  # VTK_TETRA == 10
    hex_type = tvtk.Hexahedron().cell_type  # VTK_HEXAHEDRON == 12
    cell_types = np.array([tetra_type, hex_type])
    # Create the array of cells unambiguously.
    cell_array = tvtk.CellArray()
    cell_array.set_cells(2, cells)
    # Now create the UG.
    ug = tvtk.UnstructuredGrid(points=points)
    # Now just set the cell types and reuse the ug locations and cells.
    ug.set_cells(cell_types, offset, cell_array)
    return ug


def save_xml(ug, file_name):
    """Shows how you can save the unstructured grid dataset to a VTK
    XML file."""
    # w = tvtk.XMLUnstructuredGridWriter(input=ug, file_name=file_name)
    w = tvtk.XMLUnstructuredGridWriter(file_name=file_name)
    w.set_input_data(ug)
    # kkhk, file_name=file_name)
    w.write()

    # ----------------------------------------------------------------------
    # Create the unstructured grids and assign scalars and vectors.
    ug1 = single_type_ug()
    ug2 = mixed_type_ug()
    temperature = np.arange(0, 120, 10, 'd')
    velocity = np.random.randn(12, 3)
    for ug in ug1, ug2:
        ug.point_data.scalars = temperature
        ug.point_data.scalars.name = 'temperature'
        # Some vectors.
        ug.point_data.vectors = velocity
        ug.point_data.vectors.name = 'velocity'

    # Uncomment this to save the file to a VTK XML file.
    save_xml(ug2, 'file.vtu')

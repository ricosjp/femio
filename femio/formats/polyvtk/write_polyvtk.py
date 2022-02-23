
import fileinput

import numpy as np
from tvtk.api import tvtk

from ... import fem_writer
from ... import config


class PolyVTKWriter(fem_writer.FEMWriter):

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
                config.DICT_ELEMENT_TYPE_TO_VTK_ID['polyhedron'],
                list(self.fem_data.elemental_data['face'][
                    'polyhedron'].data))

        for element_type, element_data in self.fem_data.elements.items():
            if element_type == 'polyhedron':
                continue
            element_type_id = config.DICT_ELEMENT_TYPE_TO_VTK_ID[element_type]
            for element in self.fem_data.nodes.ids2indices(element_data.data):
                unstructured_grid.insert_next_cell(element_type_id, element)

        # Add cell data
        n_element = len(self.fem_data.elements)
        elemental_data_dict_2d = self.try_convert_to_2d(mode='elemental')
        for k, v in elemental_data_dict_2d.items():
            if len(v) != n_element:
                continue
            if k == 'face':
                continue
            array_id = unstructured_grid.cell_data.add_array(
                self._reorder_cell_data(v.data))
            unstructured_grid.cell_data.get_array(array_id).name = k

        # Add point data
        n_node = len(self.fem_data.nodes)
        nodal_data_dict_2d = self.try_convert_to_2d(mode='nodal')
        for k, v in nodal_data_dict_2d.items():
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
        return file_name

    def _reorder_cell_data(self, data):
        return np.concatenate(
            [data[self.fem_data.elements.types == 'polyhedron']] + [
                data[self.fem_data.elements.types == t]
                for t, _ in self.fem_data.elements.items()
                if t != 'polyhedron'])

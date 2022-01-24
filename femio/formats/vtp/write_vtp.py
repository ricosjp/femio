
import fileinput

import numpy as np
from tvtk.api import tvtk

from ... import fem_writer
from . import vtp


class VTPWriter(fem_writer.FEMWriter):

    DICT_ELEMENT_TYPE_TO_VTK_ID = {
        v: k for k, v
        in vtp.VTPData.DICT_VTK_ID_TO_ELEMENT_TYPE.items()}

    def write(self, file_name=None, *, overwrite=False):
        """Write FEM data in VTK polydata (vtp) format.

        Parameters
        ----------
        file_name: str
            File name of the output file. If not fed,
            input_filename.out.stl will be the output file name.
        overwrite: bool, optional [False]
            If True, allow averwrite files.
        """
        poly_data = tvtk.PolyData(
            points=self.fem_data.nodes.data)
        poly_data.polys.set_cells(
            len(self.fem_data.elements), self.fem_data.elements.to_vtk(
                self.fem_data.nodes))

        # Add cell data
        n_element = len(self.fem_data.elements)
        elemental_data_dict_2d = self.try_convert_to_2d(mode='elemental')
        for k, v in elemental_data_dict_2d.items():
            if len(v) != n_element:
                continue
            array_id = poly_data.cell_data.add_array(v.data)
            poly_data.cell_data.get_array(array_id).name = k

        # Add point data
        n_node = len(self.fem_data.nodes)
        nodal_data_dict_2d = self.try_convert_to_2d(mode='nodal')
        for k, v in nodal_data_dict_2d.items():
            if len(v) != n_node:
                continue
            array_id = poly_data.point_data.add_array(v.data)
            poly_data.point_data.get_array(array_id).name = k

        # Clean up unnecessary data
        poly_data.verts = tvtk.CellArray()
        poly_data.lines = tvtk.CellArray()
        poly_data.strips = tvtk.CellArray()

        writer = tvtk.XMLPolyDataWriter(file_name=str(file_name))
        writer.data_mode = 'ascii'
        writer.set_input_data(poly_data)
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

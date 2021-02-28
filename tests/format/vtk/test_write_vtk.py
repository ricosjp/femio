import os
from pathlib import Path
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestWriteVTK(unittest.TestCase):

    def test_fistr_to_vtk(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3_node_disordered',
            read_npy=False, save=False)
        write_file_name = 'tests/data/vtk/write_from_fistr/mesh.vtk'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('vtk', write_file_name)

        vtk_fem_data = FEMData.read_files(
            'vtk', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, vtk_fem_data.nodes.data)
        np.testing.assert_almost_equal(
            np.ravel(fem_data.nodal_data.get_attribute_data('t_init')),
            np.ravel(vtk_fem_data.nodal_data.get_attribute_data('t_init')))
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('lte'),
            vtk_fem_data.elemental_data.get_attribute_data('lte'))

    def test_stl_to_vtk(self):
        fem_data = FEMData.read_files(
            'stl', ['tests/data/stl/multiple/multiple.stl'])
        write_file_name = 'tests/data/vtk/write_from_stl/mesh.vtk'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('vtk', write_file_name)

        vtk_fem_data = FEMData.read_files(
            'vtk', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, vtk_fem_data.nodes.data)

    def test_read_hexcol(self):
        file_name = Path('tests/data/vtk/hexcol/mesh.vtk')

        fem_data = FEMData.read_files('vtk', [file_name])
        write_file_name = Path('tests/data/vtk/write_hexcol/mesh.vtk')

        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('vtk', write_file_name)
        written_fem_data = FEMData.read_files('vtk', file_name)
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)
        np.testing.assert_almost_equal(
            written_fem_data.elements.data,
            fem_data.elements.data)

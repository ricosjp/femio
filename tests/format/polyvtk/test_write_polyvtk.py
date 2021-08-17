import pathlib
import shutil
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestWriteVTK(unittest.TestCase):

    def test_write_polyhedron(self):
        file_name = pathlib.Path('tests/data/vtu/polyhedron/polyhedron.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])

        write_file_name = pathlib.Path(
            'tests/data/vtu/write_polyhedron/write_polyhedron.vtu')
        if write_file_name.exists():
            shutil.rmtree(write_file_name.parent)

        fem_data.write('polyvtk', write_file_name)
        written_fem_data = FEMData.read_files('polyvtk', [write_file_name])
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        ae, de = written_fem_data.elements.data, fem_data.elements.data
        np.testing.assert_almost_equal(ae[0], [1, 2, 4, 6, 9, 10, 11, 12])
        np.testing.assert_almost_equal(ae[1], de[1])
        np.testing.assert_almost_equal(ae[2], de[2])

        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('U'),
            fem_data.elemental_data.get_attribute_data('U'))
        np.testing.assert_almost_equal(
            written_fem_data.nodal_data.get_attribute_data('X'),
            fem_data.nodal_data.get_attribute_data('X'))

    def test_write_mix_poly(self):
        file_name = pathlib.Path('tests/data/vtu/mix_poly/mix_poly.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])

        write_file_name = pathlib.Path(
            'tests/data/vtu/write_mix_poly/write_mix_poly.vtu')
        if write_file_name.exists():
            shutil.rmtree(write_file_name.parent)

        fem_data.write('polyvtk', write_file_name)
        written_fem_data = FEMData.read_files('polyvtk', [write_file_name])
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        ae, de = written_fem_data.elements.data, fem_data.elements.data
        np.testing.assert_almost_equal(ae[0], [1, 2, 4, 6, 9, 10, 11, 12])
        np.testing.assert_almost_equal(ae[1], de[2])
        np.testing.assert_almost_equal(ae[2], de[1])

        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('U'),
            fem_data.elemental_data.get_attribute_data('U')[[0, 2, 1]])
        np.testing.assert_almost_equal(
            written_fem_data.nodal_data.get_attribute_data('X'),
            fem_data.nodal_data.get_attribute_data('X'))

    def test_write_mix_poly2(self):
        file_name = pathlib.Path('tests/data/vtu/mix_poly2/mix_poly2.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])

        write_file_name = pathlib.Path(
            'tests/data/vtu/write_mix_poly2/write_mix_poly2.vtu')
        if write_file_name.exists():
            shutil.rmtree(write_file_name.parent)

        fem_data.write('polyvtk', write_file_name)
        written_fem_data = FEMData.read_files('polyvtk', [write_file_name])
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        ae, de = written_fem_data.elements.data, fem_data.elements.data
        np.testing.assert_almost_equal(ae[0], [1, 2, 4, 6, 9, 10, 11, 12])
        np.testing.assert_almost_equal(ae[1], de[1])
        np.testing.assert_almost_equal(ae[2], de[2])
        np.testing.assert_almost_equal(ae[3], de[3])

        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('U'),
            fem_data.elemental_data.get_attribute_data('U'))
        np.testing.assert_almost_equal(
            written_fem_data.nodal_data.get_attribute_data('X'),
            fem_data.nodal_data.get_attribute_data('X'))

    # def test_io_openfoam(self):
    #     directory_name = pathlib.Path('tests/data/vtu/openfoam')
    #     fem_data = FEMData.read_directory(
    #         'polyvtk', directory_name, read_npy=False, save=False)
    #     write_directory_name = pathlib.Path('tests/data/vtu/write_openfoam')
    #     if write_directory_name.exists():
    #         shutil.rmtree(write_directory_name)
    #     fem_data.write('polyvtk', write_directory_name / 'mesh.vtu')

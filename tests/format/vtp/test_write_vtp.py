import pathlib
import shutil
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestWriteVTP(unittest.TestCase):

    def test_write_vtp_closed(self):
        file_name = pathlib.Path('tests/data/vtp/closed/mesh.vtp')
        fem_data = FEMData.read_files('vtp', [file_name])

        write_file_name = pathlib.Path(
            'tests/data/vtp/write_closed/mesh.vtp')
        if write_file_name.exists():
            shutil.rmtree(write_file_name.parent)

        fem_data.write('vtp', write_file_name)
        written_fem_data = FEMData.read_files('vtp', [write_file_name])
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        for ae, de in zip(
                written_fem_data.elements.data, fem_data.elements.data):
            np.testing.assert_almost_equal(ae, np.array(de))

        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('p'),
            fem_data.elemental_data.get_attribute_data('p'))
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('U'),
            fem_data.elemental_data.get_attribute_data('U'))

        np.testing.assert_almost_equal(
            written_fem_data.nodal_data.get_attribute_data('p'),
            fem_data.nodal_data.get_attribute_data('p'))
        np.testing.assert_almost_equal(
            written_fem_data.nodal_data.get_attribute_data('U'),
            fem_data.nodal_data.get_attribute_data('U'))

    def test_write_vtp_polys(self):
        file_name = pathlib.Path('tests/data/vtp/polys/mesh.vtp')
        fem_data = FEMData.read_files('vtp', [file_name])

        write_file_name = pathlib.Path(
            'tests/data/vtp/write_polys/mesh.vtp')
        if write_file_name.exists():
            shutil.rmtree(write_file_name.parent)

        fem_data.write('vtp', write_file_name)
        written_fem_data = FEMData.read_files('vtp', [write_file_name])
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        for ae, de in zip(
                written_fem_data.elements.data, fem_data.elements.data):
            np.testing.assert_almost_equal(ae, np.array(de))

        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('p'),
            fem_data.elemental_data.get_attribute_data('p'))
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('U'),
            fem_data.elemental_data.get_attribute_data('U'))

        np.testing.assert_almost_equal(
            written_fem_data.nodal_data.get_attribute_data('p'),
            fem_data.nodal_data.get_attribute_data('p'))
        np.testing.assert_almost_equal(
            written_fem_data.nodal_data.get_attribute_data('U'),
            fem_data.nodal_data.get_attribute_data('U'))

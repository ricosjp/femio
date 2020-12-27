import pathlib
import os
import re
import shutil
import unittest

import numpy as np

from femio.fem_attribute import FEMAttribute
from femio.fem_data import FEMData


class TestWriteUCD(unittest.TestCase):

    def test_write_ucd(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/thermal/hex.msh',
            'tests/data/fistr/thermal/hex.cnt',
            'tests/data/fistr/thermal/hex.res.0.1'])
        write_file_name = 'tests/data/ucd/write/mesh.inp'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('ucd', write_file_name)

        ucd_fem_data = FEMData.read_files(
            'ucd', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('NodalSTRESS'),
            ucd_fem_data.nodal_data.get_attribute_data('NodalSTRESS'))

    def test_write_ucd_mesh_only(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/thermal/hex.msh'])
        write_file_name = 'tests/data/ucd/write/mesh_only.inp'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('ucd', write_file_name)

        ucd_fem_data = FEMData.read_files(
            'ucd', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data,
            ucd_fem_data.nodes.data)

    def test_write_ucd_heat(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat_steady/hex.msh',
            'tests/data/fistr/heat_steady/hex.cnt',
            'tests/data/fistr/heat_steady/hex.res.0.1'])
        write_file_name = 'tests/data/ucd/write_heat_steady/mesh.inp'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('ucd', write_file_name)

        # ucd_fem_data = FEMData.read_files(
        #     'ucd', write_file_name)
        # np.testing.assert_almost_equal(
        #     fem_data.nodal_data.get_attribute_data('TEMPERATURE'),
        #     ucd_fem_data.nodal_data.get_attribute_data('TEMPERATURE'))

    def test_write_ucd_quad(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/quad', read_npy=False,
            save=False)

        write_file_name = 'tests/data/ucd/write_quad/mesh.inp'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('ucd', write_file_name)

        ucd_fem_data = FEMData.read_files(
            'ucd', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data,
            ucd_fem_data.nodes.data)

    def test_write_ucd_nan(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False,
            save=False)
        fem_data.nodal_data['INITIAL_TEMPERATURE'].data[3, 0] = np.nan
        fem_data.elemental_data['Young_modulus'].data[1, 0] = np.nan

        write_file_name = 'tests/data/ucd/write_nan/mesh.inp'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('ucd', write_file_name)

        ucd_fem_data = FEMData.read_files(
            'ucd', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('INITIAL_TEMPERATURE'),
            ucd_fem_data.nodal_data.get_attribute_data('INITIAL_TEMPERATURE'))
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('Young_modulus'),
            ucd_fem_data.elemental_data.get_attribute_data('Young_modulus'))

    def test_write_ucd_mixed_solid(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid',
            read_npy=False, save=False)
        # raise ValueError(fem_data.elemental_data.get_attribute_data(
        #     'Young_modulus'))

        write_file_name = 'tests/data/ucd/write_mixture_solid/mesh.inp'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('ucd', write_file_name)

        ucd_fem_data = FEMData.read_files(
            'ucd', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('DISPLACEMENT'),
            ucd_fem_data.nodal_data.get_attribute_data('DISPLACEMENT'))
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('Young_modulus'),
            ucd_fem_data.elemental_data.get_attribute_data('Young_modulus'))

    def test_write_ucd_tet2_to_tet(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2',
            read_npy=False, save=False)

        write_file_name = 'tests/data/ucd/tet2/mesh.inp'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('ucd', write_file_name)

        tet_found = False
        with open(write_file_name, 'r') as file_:
            for line in file_:
                if re.search(' tet ', line):
                    tet_found = True
        self.assertTrue(tet_found)

    def test_write_time_series(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_tet2_3', read_npy=False,
            save=False, time_series=True)
        fem_data.elemental_data['volume'] = FEMAttribute(
            'volume', fem_data.elements.ids,
            fem_data.calculate_element_volumes())
        write_dir_name = pathlib.Path(
            'tests/data/ucd/write_heat_tet2_3')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write('ucd', write_dir_name / 'mesh.inp')

import unittest

import numpy as np

from femio.fem_attribute import FEMAttribute
from femio.fem_data import FEMData


RUN_FISTR = True


INP_FILE = 'tests/data/ucd/hex.inp'
FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'


class TestFemAttribute(unittest.TestCase):

    def test_ids2indices(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        desired_indices = [0, 3, 5, 11]
        np.testing.assert_array_equal(
            fem_data.nodes.ids2indices([1, 4, 6, 12]), desired_indices)

    def test_ids2indices_multi_dimensions(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        desired_indices = [[0, 3], [5, 11]]
        np.testing.assert_array_equal(
            fem_data.nodes.ids2indices([[1, 4], [6, 12]]), desired_indices)

    def test_femattribute_length_different_error(self):
        """ValueError should be raised when length of IDs and data differ."""
        with self.assertRaises(ValueError):
            FEMAttribute('', [1, 2, 3], [10., 20., 30., 40.])

    def test_get_attribute_nodal(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False)
        displacement = fem_data.nodal_data.get_attribute_data('displacement')
        displacement_alias = fem_data.nodal_data.get_attribute_data(
            'DISPLACEMENT')
        np.testing.assert_almost_equal(
            displacement, fem_data.nodal_data['DISPLACEMENT'].data)
        np.testing.assert_almost_equal(
            displacement_alias, fem_data.nodal_data['DISPLACEMENT'].data)

    def test_get_attribute_elemental(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False)
        lte = fem_data.elemental_data.get_attribute_data('lte')
        lte_alias = fem_data.elemental_data.get_attribute_data(
            'linear_thermal_expansion_coefficient')
        np.testing.assert_almost_equal(
            lte, fem_data.elemental_data[
                'linear_thermal_expansion_coefficient'].data)
        np.testing.assert_almost_equal(
            lte_alias, fem_data.elemental_data[
                'linear_thermal_expansion_coefficient'].data)

    def test_overwrite_attribute(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False)
        data = np.random.rand(
            *fem_data.elemental_data.get_attribute_data('lte').shape)
        fem_data.elemental_data['lte'].data = data
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('lte'), data)

    def test_time_series_interfaces(self):
        data = np.reshape(np.arange(10. * 5 * 3), (10, 5, 3))
        time_series = FEMAttribute(
            'TIME_SERIES', ids=np.arange(5) + 1, data=data, time_series=True)
        np.testing.assert_array_equal(time_series.ids, [1, 2, 3, 4, 5])

        np.testing.assert_almost_equal(
            time_series.iloc[2].values, data[:, [2], :])
        np.testing.assert_almost_equal(
            time_series.iloc[[1, 3]].values, data[:, [1, 3], :])
        np.testing.assert_almost_equal(
            time_series.iloc[1:3].values, data[:, 1:3, :])

        np.testing.assert_almost_equal(
            time_series.loc[2].values, data[:, [1], :])
        np.testing.assert_almost_equal(
            time_series.loc[[1, 3]].values, data[:, [0, 2], :])
        np.testing.assert_almost_equal(
            time_series.loc[1:3].values, data[:, 0:3, :])

        twice_data = data * 2
        time_series.data = twice_data

        np.testing.assert_almost_equal(
            time_series.iloc[2].values, twice_data[:, [2], :])
        np.testing.assert_almost_equal(
            time_series.iloc[[1, 3]].values, twice_data[:, [1, 3], :])
        np.testing.assert_almost_equal(
            time_series.iloc[1:3].values, twice_data[:, 1:3, :])

        np.testing.assert_almost_equal(
            time_series.loc[2].values, twice_data[:, [1], :])
        np.testing.assert_almost_equal(
            time_series.loc[[1, 3]].values, twice_data[:, [0, 2], :])
        np.testing.assert_almost_equal(
            time_series.loc[1:3].values, twice_data[:, 0:3, :])

    def test_fem_attribute_multidimension(self):
        a = np.reshape(np.arange(5 * 3 * 3), (5, 3, 3))
        ids = ['A', 'B', 'C', 'D', 'E']
        fem_attribute = FEMAttribute('data', ids=ids, data=a)
        np.testing.assert_array_equal(
            fem_attribute.loc[['B', 'D']].data, a[[1, 3]])
        np.testing.assert_array_equal(
            fem_attribute.loc['C'].data, a[[2]])
        np.testing.assert_array_equal(
            fem_attribute.iloc[[1, 3]].data, a[[1, 3]])
        np.testing.assert_array_equal(
            fem_attribute.iloc[2].data, a[[2]])

        fem_attribute.loc[['E']].data = 100 * np.ones((1, 3, 3))
        np.testing.assert_array_equal(
            fem_attribute.loc['E'].data, 100 * np.ones((1, 3, 3)))

    def test_update_elemental_data(self):
        fem_data = FEMData.read_directory("obj", "tests/data/obj/quad")
        new_ids = np.array([3, 5])
        new_data = np.array([
            [1, 3, 6, 4],
            [3, 7, 8, 6]
        ])
        new_quad = FEMAttribute('quad', ids=new_ids, data=new_data)
        fem_data.elements.update({'quad': new_quad})

        np.testing.assert_array_equal(fem_data.elements.ids, np.array([3, 5]))
        np.testing.assert_array_equal(
            fem_data.elements.types, np.array(['quad', 'quad']))

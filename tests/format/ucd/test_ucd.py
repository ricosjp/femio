import unittest

import numpy as np

from femio.fem_data import FEMData


class TestFEMDataFISTR(unittest.TestCase):

    def test_ucd_nodes(self):
        data_directory = 'tests/data/ucd/thermal'
        fem_data = FEMData.read_directory(
            'ucd', data_directory, read_npy=False)

        desired_node_ids = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)

        desired_node_positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_positions)

    def test_ucd_elements(self):
        data_directory = 'tests/data/ucd/thermal'
        fem_data = FEMData.read_directory(
            'ucd', data_directory, read_npy=False)

        desired_element_ids = np.array([1, 2])
        np.testing.assert_array_equal(
            fem_data.elements.ids, desired_element_ids)

        desired_element_data = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9, 10, 11, 12],
        ])
        np.testing.assert_array_equal(
            fem_data.elements.data, desired_element_data)

    def test_ucd_nodal_data(self):
        data_directory = 'tests/data/ucd/thermal'
        fem_data = FEMData.read_directory(
            'ucd', data_directory, read_npy=False)
        desired_initial_temperatures = np.array([
            [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.]]).T
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('INITIAL_TEMPERATURE'),
            desired_initial_temperatures)

    def test_ucd_elemental_data(self):
        data_directory = 'tests/data/ucd/thermal'
        fem_data = FEMData.read_directory(
            'ucd', data_directory, read_npy=False)
        desired_lte = np.array([
            [.001, .002, .003],
            [-.001, .004, .019],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('lte'),
            desired_lte)

    def test_ucd_spaces(self):
        data_directory = 'tests/data/ucd/spaces'
        fem_data = FEMData.read_directory(
            'ucd', data_directory, read_npy=False)

        desired_node_ids = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)

        desired_node_positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_positions)

        desired_nodal_mises = np.array([[
            1.0898350e+04,
            1.3781944e+04,
            1.4294185e+04,
            1.8647205e+04,
            9.3792805e+03,
            7.3153108e+03,
            6.7785976e+03,
            5.1787984e+03,
            7.4483455e+01,
            6.4159305e+01,
            4.9329406e+01,
            2.9528125e+01,
        ]]).T
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('NodalMISES'),
            desired_nodal_mises)

    def test_ucd_nans(self):
        data_directory = 'tests/data/ucd/nan'
        fem_data = FEMData.read_directory(
            'ucd', data_directory, read_npy=False)
        desired_initial_temperatures = np.array([
            [10., 20., 30., np.nan, 50., 60., 70.,
             80., 90., 100., 110., 120.]]).T
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('INITIAL_TEMPERATURE'),
            desired_initial_temperatures)

        desired_young_modululi = np.array([[210000., np.nan]]).T
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('Young_modulus'),
            desired_young_modululi)

    def test_read_files_read_directory(self):
        data_directory = 'tests/data/ucd/thermal'
        fem_data_dir = FEMData.read_directory(
            'ucd', data_directory, read_npy=False)
        fem_data_file = FEMData.read_files(
            'ucd', data_directory + '/mesh.inp')
        np.testing.assert_almost_equal(
            fem_data_dir.elemental_data.get_attribute_data('ElementalSTRAIN'),
            fem_data_file.elemental_data.get_attribute_data('ElementalSTRAIN'))

    def test_read_mixture(self):
        data_directory = 'tests/data/ucd/mixture'
        fem_data = FEMData.read_directory(
            'ucd', data_directory, read_npy=False, save=False)
        desired_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)
        np.testing.assert_array_equal(
            fem_data.elements['hex'].data,
            np.array([
                [1, 2, 4, 3, 5, 6, 8, 7],
                [13, 3, 4, 14, 15, 7, 8, 16],
            ]))
        np.testing.assert_array_equal(
            fem_data.elements['tet'].data,
            np.array([
                [2, 12, 6, 9]
            ]))
        np.testing.assert_array_equal(
            fem_data.elements['quad'].data,
            np.array([
                [6, 9, 10, 8],
            ]))
        np.testing.assert_array_equal(
            fem_data.elements['tri'].data,
            np.array([
                [9, 11, 10],
            ]))

    def test_read_no_nodal(self):
        data_directory = 'tests/data/ucd/no_nodal'
        FEMData.read_directory(
            'ucd', data_directory, read_npy=False, save=False)

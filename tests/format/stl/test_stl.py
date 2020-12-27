import unittest

import numpy as np

from femio.fem_data import FEMData


class TestFEMDataSTL(unittest.TestCase):

    def test_stl(self):
        data_directory = 'tests/data/stl/simple'
        fem_data = FEMData.read_directory(
            'stl', data_directory, read_npy=False, save=False)

        desired_node_ids = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)

        desired_node_positions = np.array([
            [0., 0., 0.],
            [.2, 3., 0.],
            [2., .1, 0.],
            [4., 5., 0.],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_positions)

        desired_element_ids = np.array([1, 2])
        np.testing.assert_array_equal(
            fem_data.elements.ids, desired_element_ids)
        np.testing.assert_array_equal(
            fem_data.elements.data,
            np.array([
                [1, 3, 2],
                [4, 2, 3],
            ]))

    def test_stl_color(self):
        data_directory = 'tests/data/stl/simple_w_color'
        fem_data = FEMData.read_directory(
            'stl', data_directory, read_npy=False, save=False)

        desired_node_ids = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)

        desired_node_positions = np.array([
            [0., 0., 0.],
            [.2, 3., 0.],
            [2., .1, 0.],
            [4., 5., 0.],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_positions)

        desired_element_ids = np.array([1, 2])
        np.testing.assert_array_equal(
            fem_data.elements.ids, desired_element_ids)
        np.testing.assert_array_equal(
            fem_data.elements.data,
            np.array([
                [1, 3, 2],
                [4, 2, 3],
            ]))

    def test_stl_multiple(self):
        data_directory = 'tests/data/stl/multiple'
        fem_data = FEMData.read_directory(
            'stl', data_directory, read_npy=False, save=False)

        desired_node_ids = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)

        desired_node_positions = np.array([
            [0., 0., 0.],
            [.2, 3., 0.],
            [1., 0., 1.],
            [2., 0., 0.],
            [4., 5., 0.],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_positions)

        desired_element_ids = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            fem_data.elements.ids, desired_element_ids)
        np.testing.assert_array_equal(
            fem_data.elements.data,
            np.array([
                [1, 4, 2],
                [5, 2, 4],
                [1, 4, 3],
            ]))

    def test_stl_binary(self):
        data_directory = 'tests/data/stl/binary'
        fem_data = FEMData.read_directory(
            'stl', data_directory, read_npy=False, save=False)
        desired_node_ids = np.array([1, 2, 3, 4, 5, 6, 7])
        desired_node_data = np.array([
            [0., 0., 0.],
            [0., 0., 2.],
            [0., 0., 4.],
            [0., 2., 2.],
            [0., 4., 0.],
            [2., 0., 2.],
            [4., 0., 0.]])
        desired_element_data = np.array([
            [7, 5, 1],
            [7, 5, 4],
            [7, 1, 2],
            [7, 6, 4],
            [7, 6, 2],
            [5, 1, 2],
            [5, 4, 2],
            [3, 6, 4],
            [3, 6, 2],
            [3, 4, 2]])

        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_data)
        np.testing.assert_array_equal(
            fem_data.elements.data,
            desired_element_data)

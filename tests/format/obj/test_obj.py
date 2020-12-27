import unittest

import numpy as np

from femio.fem_data import FEMData


class TestFEMDataObj(unittest.TestCase):

    def test_obj(self):
        data_directory = 'tests/data/obj/mixture_shell'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_node_ids = np.array(
                [1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)

        desired_node_positions = np.array([
            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
            [1.0000000e+00, 0.0000000e+00, 0.0000000e+00],
            [1.0000000e+00, 1.0000000e+00, 0.0000000e+00],
            [0.0000000e+00, 1.0000000e+00, 0.0000000e+00],
            [2.0000000e+00, 0.0000000e+00, 1.0000000e+00],
            [1.0000000e+00, 1.0000000e+00, 1.0000000e+00],
            [0.0000000e+00, 1.0000000e+00, 1.0000000e+00],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_positions)

        desired_element_ids = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            fem_data.elements.ids, desired_element_ids)
        np.testing.assert_array_equal(
            fem_data.elements['tri'].data,
            np.array([
                [2, 3, 5],
            ]))
        np.testing.assert_array_equal(
            fem_data.elements['quad'].data,
            np.array([
                [1, 2, 3, 4],
                [3, 6, 7, 4],
            ]))

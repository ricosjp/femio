import os
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestWriteSTL(unittest.TestCase):

    def test_write_stl(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3', read_npy=False)
        write_file_name = 'tests/data/stl/write/out.stl'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('stl', write_file_name)
        self.assertTrue(os.path.exists(write_file_name))

        fem_data = FEMData.read_files(
            'stl', write_file_name)
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
            [7, 1, 5],
            [7, 5, 4],
            [7, 2, 1],
            [7, 4, 6],
            [7, 6, 2],
            [5, 1, 2],
            [5, 2, 4],
            [6, 4, 3],
            [6, 3, 2],
            [4, 2, 3]])

        np.testing.assert_array_equal(
            fem_data.nodes.ids, desired_node_ids)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_data)
        np.testing.assert_array_equal(
            fem_data.elements.data,
            desired_element_data)

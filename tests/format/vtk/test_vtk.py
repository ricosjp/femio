from pathlib import Path
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestVTK(unittest.TestCase):

    def test_read_vtk_tet2(self):
        file_name = Path('tests/data/vtk/tet2_3/mesh.vtk')
        fem_data = FEMData.read_files('vtk', [file_name])

        desired_nodes = np.array([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [5, 6, 7, 4, 8, 9, 10, 11, 12, 13],
            [1, 2, 3, 7, 14, 15, 16, 17, 18, 19],
            [1, 6, 7, 5, 8, 17, 20, 21, 10, 9],
            [1, 2, 7, 6, 18, 17, 16, 20, 22, 8],
        ])
        np.testing.assert_almost_equal(
            fem_data.elements.data, desired_elements)

    def test_read_vtk(self):
        file_name = Path('tests/data/vtk/tet_3/mesh.vtk')
        fem_data = FEMData.read_files('vtk', [file_name])

        desired_nodes = np.array([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [5, 6, 7, 4],
            [1, 2, 3, 7],
            [1, 6, 7, 5],
            [1, 2, 7, 6],
        ])
        np.testing.assert_almost_equal(
            fem_data.elements.data, desired_elements)

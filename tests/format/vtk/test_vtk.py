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

    def test_read_hex(self):
        file_name = Path('tests/data/vtk/hex/mesh.vtk')

        fem_data = FEMData.read_files('vtk', [file_name])
        desired_nodes = np.array([
            [0.0, 0, 0],
            [0.1, 0, 0],
            [0.1, 0.1, 0],
            [0.0, 0.1, 0],
            [0.0, 0, 0.005],
            [0.1, 0, 0.005],
            [0.1, 0.1, 0.005],
            [0.0, 0.1, 0.005],
            [0.0, 0, 0.01],
            [0.1, 0, 0.01],
            [0.1, 0.1, 0.01],
            [0.0, 0.1, 0.01],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9, 10, 11, 12],
        ])
        np.testing.assert_almost_equal(
            fem_data.elements.data, desired_elements)
        self.assertEqual(fem_data.elements.element_type, 'hex')
        np.testing.assert_almost_equal(
            fem_data.calculate_element_volumes(),
            np.ones((2, 1)) * .1 * .1 * .005)

    def test_read_hexprism(self):
        file_name = Path('tests/data/vtk/hexprism/mesh.vtk')

        fem_data = FEMData.read_files('vtk', [file_name])
        desired_nodes = np.array([
            [0.0, 0, 0],
            [0.1, 0, 0],
            [0.1, 0.1, 0],
            [0.06, 0.1, 0],
            [0.03, 0.1, 0],
            [0.0, 0.1, 0],
            [0.0, 0, 0.005],
            [0.1, 0, 0.005],
            [0.1, 0.1, 0.005],
            [0.06, 0.1, 0.005],
            [0.03, 0.1, 0.005],
            [0.0, 0.1, 0.005],
            [0.0, 0, 0.01],
            [0.1, 0, 0.01],
            [0.1, 0.1, 0.01],
            [0.06, 0.1, 0.01],
            [0.03, 0.1, 0.01],
            [0.0, 0.1, 0.01],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        ])
        np.testing.assert_almost_equal(
            fem_data.elements.data, desired_elements)
        self.assertEqual(fem_data.elements.element_type, 'hexprism')
        np.testing.assert_almost_equal(
            fem_data.calculate_element_volumes(),
            np.ones((2, 1)) * .1 * .1 * .005)

    def test_read_mix_hex_hexprism(self):
        file_name = Path('tests/data/vtk/mix_hex_hexprism/mesh.vtk')

        fem_data = FEMData.read_files('vtk', [file_name])
        desired_nodes = np.array([
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.3, 0.0, 0.3],
            [0.6, 0.0, 0.3],
            [0.9, 0.0, 0.3],
            [0.4, 0.0, 0.4],
            [0.5, 0.0, 0.4],
            [0.4, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 0.6],
            [0.3, 0.0, 0.6],
            [0.6, 0.0, 0.6],
            [0.9, 0.0, 0.6],
            [0.0, 0.0, 0.9],
            [0.3, 0.0, 0.9],
            [0.6, 0.0, 0.9],
            [0.9, 0.0, 0.9],
            [0.0, 0.1, 0.0],
            [0.3, 0.1, 0.0],
            [0.6, 0.1, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.1, 0.3],
            [0.3, 0.1, 0.3],
            [0.6, 0.1, 0.3],
            [0.9, 0.1, 0.3],
            [0.4, 0.1, 0.4],
            [0.5, 0.1, 0.4],
            [0.4, 0.1, 0.5],
            [0.5, 0.1, 0.5],
            [0.0, 0.1, 0.6],
            [0.3, 0.1, 0.6],
            [0.6, 0.1, 0.6],
            [0.9, 0.1, 0.6],
            [0.0, 0.1, 0.9],
            [0.3, 0.1, 0.9],
            [0.6, 0.1, 0.9],
            [0.9, 0.1, 0.9],
            [-0.1, 0.2, -0.1],
            [0.3, 0.2, -0.1],
            [0.6, 0.2, -0.1],
            [1.0, 0.2, -0.1],
            [-0.1, 0.2, 0.3],
            [0.3, 0.2, 0.3],
            [0.6, 0.2, 0.3],
            [1.0, 0.2, 0.3],
            [0.4, 0.2, 0.4],
            [0.5, 0.2, 0.4],
            [0.4, 0.2, 0.5],
            [0.5, 0.2, 0.5],
            [-0.1, 0.2, 0.6],
            [0.3, 0.2, 0.6],
            [0.6, 0.2, 0.6],
            [1.0, 0.2, 0.6],
            [-0.1, 0.2, 1.0],
            [0.3, 0.2, 1.0],
            [0.6, 0.2, 1.0],
            [1.0, 0.2, 1.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        self.assertEqual(fem_data.elements.element_type, 'mix')
        np.testing.assert_almost_equal(
            fem_data.calculate_element_volumes(),
            np.ones((2, 1)) * .1 * .1 * .005)

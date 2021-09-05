import pathlib
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestVTP(unittest.TestCase):

    def test_read_vtp_closed(self):
        file_name = pathlib.Path('tests/data/vtp/closed/mesh.vtp')
        fem_data = FEMData.read_files('vtp', [file_name])

        desired_nodes = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.5, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.5, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [0, 1, 2, 3, 4],
            [0, 4, 5],
            [5, 4, 9],
            [4, 3, 9],
            [3, 8, 9],
            [3, 2, 8],
            [2, 7, 8],
            [1, 6, 7, 2],
            [0, 5, 6, 1],
            [5, 9, 8, 6],
            [6, 8, 7],
        ], dtype=object)
        for ae, de in zip(fem_data.elements.data, desired_elements):
            np.testing.assert_almost_equal(ae, np.array(de) + 1)

        desired_elemental_p = np.array([
            [0.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.5],
            [2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('p'),
            desired_elemental_p)

        desired_elemental_u = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('U'),
            desired_elemental_u)

        desired_nodal_p = np.array([
            [0.0],
            [1.0],
            [1.5],
            [1.0],
            [0.0],
            [1.0],
            [2.0],
            [2.5],
            [2.0],
            [1.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('p'),
            desired_nodal_p)

        desired_nodal_u = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.5],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.5, 1.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('U'),
            desired_nodal_u)

        desired_nodal_adj = np.array([
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
        ])
        np.testing.assert_array_equal(
            fem_data.calculate_adjacency_matrix_node().toarray().astype(int),
            desired_nodal_adj)

        desired_elemental_adj = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ])
        np.testing.assert_array_equal(
            fem_data.calculate_adjacency_matrix_element().toarray()
            .astype(int),
            desired_elemental_adj)

    def test_read_vtp_polys(self):
        file_name = pathlib.Path('tests/data/vtp/polys/mesh.vtp')
        fem_data = FEMData.read_files('vtp', [file_name])

        desired_nodes = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.5, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.5, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [-0.2, 0.0, 0.2],
            [-0.2, 0.0, 0.8],
            [1.0, 1.5, 0.5],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [0, 1, 2, 3, 4],
            [5, 9, 8, 7, 6],
            [0, 4, 9, 5, 11, 10],
            [4, 3, 12, 8, 9],
        ], dtype=object)
        for ae, de in zip(fem_data.elements.data, desired_elements):
            np.testing.assert_almost_equal(ae, np.array(de) + 1)

        desired_elemental_p = np.array([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('p'),
            desired_elemental_p)

        desired_elemental_u = np.array([
            [0.0, 0.0, 3.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 1.0],
            [0.0, 3.0, 0.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('U'),
            desired_elemental_u)

        desired_nodal_p = np.array([
            [0.0],
            [1.0],
            [1.5],
            [1.0],
            [0.0],
            [1.0],
            [2.0],
            [2.5],
            [2.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('p'),
            desired_nodal_p)

        desired_nodal_u = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.5],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 1.5],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [-0.2, 0.2, 0.0],
            [-0.2, 0.8, 0.0],
            [1.0, 0.5, 1.5],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('U'),
            desired_nodal_u)

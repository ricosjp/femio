import unittest

import numpy as np

from femio.fem_data import FEMData


RUN_FISTR = True


FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'


class TestGraphProcessor(unittest.TestCase):

    def test_calculate_adjacency_materix_node(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1', read_npy=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_node()
        desired = np.array([
            [1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

    def test_calculate_adjacency_matrix_node_tet2(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet2', read_npy=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_node()
        desired = np.array([
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

    def test_calculate_adjacency_matrix_node_mixture(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid', read_npy=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_node()
        desired = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

    def test_calculate_adjacency_matrix_node_line(self):
        fem_data = FEMData.read_directory(
            'ucd', 'tests/data/ucd/line', read_npy=False, save=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_node()
        desired = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

    def test_calculate_adjacency_materix_node_obj(self):
        fem_data = FEMData.read_directory(
            'obj', 'tests/data/obj/mixture_graph', read_npy=False, save=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_node()
        desired = np.array([
            [1, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

        elemental_adjacency_matrix = \
            fem_data.calculate_adjacency_matrix_element()
        elemental_desired = np.array([
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(
            elemental_adjacency_matrix.toarray(), elemental_desired)

    def test_calculate_element_degree(self):
        data_directory = 'tests/data/fistr/graph_tet1'
        fem_data = FEMData.read_directory(
            'fistr', data_directory, read_npy=False)
        desired_degree = np.array([
            [4],
            [4],
            [4],
            [3],
            [3]
        ], dtype=np.int64)
        fem_data.calculate_element_degree()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('degree'),
            desired_degree)

    def test_calculate_adjacency_matrix_element(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1', read_npy=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_element()
        desired = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1]], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

    def test_calculate_adjacency_matrix_element_tet2(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet2', read_npy=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_element()
        desired = np.array([
            [1, 1],
            [1, 1]], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

    def test_calculate_adjacency_matrix_element_mixture(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid', read_npy=False)
        adjacency_matrix = fem_data.calculate_adjacency_matrix_element()
        desired = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 1],
        ], dtype=bool)
        np.testing.assert_array_equal(adjacency_matrix.toarray(), desired)

    def test_extract_surface_tet(self):
        tet_data = FEMData.read_files(
            'fistr', ['tests/data/fistr/tet/tet.msh'])
        ids, _ = tet_data.extract_surface()
        desired = [[0, 2, 1],
                   [0, 1, 3],
                   [0, 3, 2],
                   [1, 2, 4],
                   [1, 4, 3],
                   [2, 3, 4]]
        np.testing.assert_array_equal(ids, desired)

    def test_extract_surface_tet_normal_correct(self):
        tet_data = FEMData.read_files(
            'fistr', ['tests/data/fistr/tet_volume/tet.msh'])
        ids, _ = tet_data.extract_surface()

        desired = np.array([
            [0, 2, 1],
            [0, 1, 3],
            [2, 0, 5],
            [0, 3, 5],
            [1, 2, 4],
            [1, 4, 3],
            [2, 5, 4],
            [3, 4, 5],
        ])
        np.testing.assert_array_equal(ids, desired)

    def test_extract_surface_hex(self):
        hex_data = FEMData.read_files(
            'fistr', ['tests/data/fistr/hex/hex.msh'])
        surface_indices, _ = hex_data.extract_surface()
        desired = [
            [0, 3, 4, 1],
            [0, 1, 10, 9],
            [3, 0, 9, 12],
            [1, 4, 5, 2],
            [1, 2, 11, 10],
            [2, 5, 14, 11],
            [3, 6, 7, 4],
            [6, 3, 12, 15],
            [4, 7, 8, 5],
            [5, 8, 17, 14],
            [7, 6, 15, 16],
            [8, 7, 16, 17],
            [9, 10, 19, 18],
            [12, 9, 18, 21],
            [10, 11, 20, 19],
            [11, 14, 23, 20],
            [15, 12, 21, 24],
            [14, 17, 26, 23],
            [16, 15, 24, 25],
            [17, 16, 25, 26],
            [18, 21, 22, 19],
            [19, 22, 23, 20],
            [21, 24, 25, 22],
            [22, 25, 26, 23]]
        np.testing.assert_array_equal(surface_indices, desired)

    def test_filter_first_order_nodes(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3', read_npy=False)
        filter_ = fem_data.filter_first_order_nodes()
        desired_nodes = [1, 2, 3, 4, 5, 6, 7]
        np.testing.assert_array_equal(
            fem_data.nodes.ids[filter_], desired_nodes)

    def test_filter_first_order_nodes_mixture(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid_2', read_npy=False)
        filter_ = fem_data.filter_first_order_nodes()
        desired_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 21, 22]
        np.testing.assert_array_equal(
            fem_data.nodes.ids[filter_], desired_nodes)

    def test_filter_first_order_nodes_disordered(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3_node_disordered', read_npy=False)
        filter_ = fem_data.filter_first_order_nodes()
        desired_nodes = [1, 2, 3, 14, 5, 6, 7]  # Respecting the order
        np.testing.assert_array_equal(
            fem_data.nodes.ids[filter_], desired_nodes)

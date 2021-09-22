import pathlib
import unittest

import numpy as np
import networkx as nx

from femio.fem_data import FEMData


class TestGraphProcessor(unittest.TestCase):

    def test_calculate_adjacency_matrix_node(self):
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

    def test_calculate_adjacency_matrix_node_obj(self):
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

    def test_calculate_adjacency_matrix_element_hex_hexprism(self):
        data_directory = pathlib.Path('tests/data/vtk/mix_hex_hexprism')
        fem_data = FEMData.read_directory(
            'vtk', data_directory, save=False, read_npy=False)
        adj = fem_data.calculate_adjacency_matrix_element()
        desired_adj = {
            0: [0, 5, 10, 11, 14, 15],
            1: [1, 6, 10, 12, 14, 16],
            2: [2, 7, 10, 11, 12, 13, 14, 15, 16, 17],
            3: [3, 8, 11, 13, 15, 17],
            4: [4, 9, 12, 13, 16, 17],
            5: [0, 5, 10, 11, 14, 15],
            6: [1, 6, 10, 12, 14, 16],
            7: [2, 7, 10, 11, 12, 13, 14, 15, 16, 17],
            8: [3, 8, 11, 13, 15, 17],
            9: [4, 9, 12, 13, 16, 17],
            10: [0, 1, 2, 5, 6, 7, 10, 11, 12, 14, 15, 16],
            11: [0, 2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 17],
            12: [1, 2, 4, 6, 7, 9, 10, 12, 13, 14, 16, 17],
            13: [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 16, 17],
            14: [0, 1, 2, 5, 6, 7, 10, 11, 12, 14, 15, 16],
            15: [0, 2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 17],
            16: [1, 2, 4, 6, 7, 9, 10, 12, 13, 14, 16, 17],
            17: [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 16, 17],
        }
        np.testing.assert_array_equal(
            adj.toarray().astype(int),
            nx.convert_matrix.to_numpy_array(
                nx.from_dict_of_lists(desired_adj)).astype(int))

    def test_calculate_adjacency_matrix_node_hex_hexprism(self):
        data_directory = pathlib.Path('tests/data/vtk/mix_hex_hexprism')
        fem_data = FEMData.read_directory(
            'vtk', data_directory, save=False, read_npy=False)
        adj = fem_data.calculate_adjacency_matrix_node()
        desired_adj = {
            0: [
                0, 1, 4, 5,
                20, 21, 24, 25,
            ],
            1: [
                0, 1, 2, 4, 5, 6, 8, 9,
                20, 21, 22, 24, 25, 26, 28, 29,
            ],
            2: [
                1, 2, 3, 5, 6, 7, 8, 9,
                21, 22, 23, 25, 26, 27, 28, 29,
            ],
            3: [
                2, 3, 6, 7,
                22, 23, 26, 27,
            ],
            4: [
                0, 1, 4, 5, 8, 10, 12, 13,
                20, 21, 24, 25, 28, 30, 32, 33,
            ],
            5: [
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13,
                20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33,
            ],
            30: [
                4, 5, 8, 9, 10, 11, 12, 13, 14, 17, 18,
                24, 25, 28, 29, 30, 31, 32, 33, 34, 37, 38,
                44, 45, 48, 49, 50, 51, 52, 53, 54, 57, 58,
            ],
        }  # Not every point is tested due to too large graph
        for start_point, end_points in desired_adj.items():
            print(f"start_point: {start_point}")
            np.testing.assert_array_equal(
                np.sort(adj[start_point, :].tocoo().col), end_points)
            np.testing.assert_array_equal(
                np.sort(adj[:, start_point].tocoo().row), end_points)

    def test_calculate_n_hop_adj(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_long',
            read_npy=False, read_mesh_only=True, save=False)
        adj1 = fem_data.calculate_n_hop_adj(
            'elemental', n_hop=1, include_self_loop=False).toarray()
        adj2 = fem_data.calculate_n_hop_adj(
            'elemental', n_hop=2, include_self_loop=False).toarray()
        adj3 = fem_data.calculate_n_hop_adj(
            'elemental', n_hop=3, include_self_loop=True).toarray()

        expected_adj1 = np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ])
        np.testing.assert_array_almost_equal(adj1, expected_adj1)

        expected_adj2 = np.array([
            [0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0],
        ])
        np.testing.assert_array_almost_equal(adj2, expected_adj2)

        expected_adj3 = np.array([
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ])
        np.testing.assert_array_almost_equal(adj3, expected_adj3)

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
        desired = np.array([
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
            [18, 19, 22, 21],
            [19, 20, 23, 22],
            [21, 22, 25, 24],
            [22, 23, 26, 25]])
        np.testing.assert_array_equal(surface_indices, desired)

    def test_to_facets(self):
        fem_data = FEMData.read_files(
            'vtk', ['tests/data/vtk/hex/mesh.vtk'])
        facet_fem_data = fem_data.to_facets()
        write_facet = pathlib.Path('tests/data/ucd/write_hex_facet/mesh.inp')
        if write_facet.exists():
            write_facet.unlink()
        facet_fem_data.write('ucd', write_facet)
        desired_facets = np.array([
            [1, 4, 3, 2],
            [1, 2, 6, 5],
            [4, 1, 5, 8],
            [2, 3, 7, 6],
            [3, 4, 8, 7],
            [5, 6, 7, 8],
            [5, 6, 10, 9],
            [8, 5, 9, 12],
            [6, 7, 11, 10],
            [7, 8, 12, 11],
            [9, 10, 11, 12]])
        np.testing.assert_array_equal(
            facet_fem_data.elements.data, desired_facets)

    def test_relative_incidence_hex(self):
        fem_data = FEMData.read_files(
            'vtk', ['tests/data/vtk/hex/mesh.vtk'])
        facet_fem_data = fem_data.to_facets()
        inc_facet2cell = fem_data.calculate_relative_incidence_metrix_element(
            facet_fem_data, minimum_n_sharing=3)
        desired_inc = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(
            inc_facet2cell.toarray().astype(int), desired_inc)

    def test_relative_incidence_graph_tet1(self):
        fem_data = FEMData.read_files(
            'fistr', ['tests/data/fistr/graph_tet1/mesh.msh'])
        facet_fem_data = fem_data.to_facets()
        inc_facet2cell = fem_data.calculate_relative_incidence_metrix_element(
            facet_fem_data, minimum_n_sharing=3)
        desired_inc = np.array([
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ])
        np.testing.assert_array_equal(
            inc_facet2cell.toarray().astype(int), desired_inc)

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

    def test_extract_surface_fistr(self):
        file_name = pathlib.Path('tests/data/fistr/tet/tet.msh')
        fem_data = FEMData.read_files('fistr', [file_name])
        surfs = fem_data.extract_surface_fistr()

        expected = np.array([1, 1, 1, 2, 1, 4, 2, 2, 2, 4, 2, 3]).reshape(6, 2)
        np.testing.assert_equal(surfs, expected)

    def test_separate(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_separate',
            read_npy=False, save=False)
        list_fem_data = fem_data.separate()

        for i, fd in enumerate(list_fem_data):
            if i == 0:
                np.testing.assert_array_equal(
                    fd.nodes.ids, [
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17])
                np.testing.assert_almost_equal(
                    fd.nodal_data.get_attribute_data('t_init')[:, 0], [
                        1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        2., 2., 2., 2., 2., 2.])
                np.testing.assert_array_equal(
                    fd.elements['hex'].data, [
                        [1, 2, 4, 3, 5, 6, 8, 7],
                        [13, 3, 4, 14, 15, 7, 8, 16]])
                np.testing.assert_array_equal(
                    fd.elements['tet'].data, [
                        [12, 2, 6, 9]])
                np.testing.assert_array_equal(
                    fd.elements['prism'].data, [
                        [1, 5, 17, 2, 6, 12]])

            elif i == 1:
                np.testing.assert_array_equal(
                    fd.nodes.ids, [
                        101, 102, 103, 104, 105, 106, 107, 108, 109, 112, 113,
                        114, 115, 116, 117,
                    ])
                np.testing.assert_almost_equal(
                    fd.nodal_data.get_attribute_data('t_init')[:, 0], [
                        3., 3., 3., 3., 3., 3., 3., 3., 3.,
                        4., 4., 4., 4., 4., 4.])
                np.testing.assert_array_equal(
                    fd.elements['hex'].data, [
                        [101, 102, 104, 103, 105, 106, 108, 107],
                        [113, 103, 104, 114, 115, 107, 108, 116]])
                np.testing.assert_array_equal(
                    fd.elements['tet'].data, [
                        [112, 102, 106, 109]])
                np.testing.assert_array_equal(
                    fd.elements['prism'].data, [
                        [101, 105, 117, 102, 106, 112]])

            else:
                raise ValueError('Separation failed')

    def test_calculate_euclidean_hop_graph_nodal(self):
        fem_data = FEMData.read_directory(
            'obj', 'tests/data/obj/mixture_plane', read_npy=False, save=False)

        adj = fem_data.calculate_euclidean_hop_graph(0.8, mode='nodal')
        desired = np.zeros((12, 12), bool)
        np.testing.assert_array_equal(adj.todense(), desired)

        adj = fem_data.calculate_euclidean_hop_graph(1.0, mode='nodal')
        desired = np.array([
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        ], dtype=bool)
        np.testing.assert_array_equal(adj.todense(), desired)

        adj = fem_data.calculate_euclidean_hop_graph(1.5, mode='nodal')
        desired = np.array([
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        ], dtype=bool)
        np.testing.assert_array_equal(adj.todense(), desired)

        adj = fem_data.calculate_euclidean_hop_graph(2.0, mode='nodal')
        desired = np.array([
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        ], dtype=bool)
        np.testing.assert_array_equal(adj.todense(), desired)

    def test_calculate_euclidean_hop_graph_elemental(self):
        fem_data = FEMData.read_directory(
            'obj', 'tests/data/obj/mixture_plane', read_npy=False, save=False)

        adj = fem_data.calculate_euclidean_hop_graph(0.8, mode='elemental')
        desired = np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ], dtype=bool)
        np.testing.assert_array_equal(adj.todense(), desired)

        adj = fem_data.calculate_euclidean_hop_graph(1.0, mode='elemental')
        desired = np.array([
            [0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 1, 0],
        ], dtype=bool)
        np.testing.assert_array_equal(adj.todense(), desired)

        adj = fem_data.calculate_euclidean_hop_graph(1.5, mode='elemental')
        desired = np.array([
            [0, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0],
        ], dtype=bool)
        np.testing.assert_array_equal(adj.todense(), desired)

    def test_nearest_neighbor_search_from_nodes_to_nodes(self):
        fem_data_1 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri', read_npy=False, save=False)

        indices, vectors = \
            fem_data_1.nearest_neighbor_search_from_nodes_to_nodes(
                2, target_fem_data=fem_data_1
            )
        desired_indices = np.array([0, 3, 1, 0, 2, 0, 3, 0]).reshape(4, 2)
        np.testing.assert_array_equal(indices, desired_indices)

        indices, vectors = \
            fem_data_1.nearest_neighbor_search_from_nodes_to_nodes(
                5, target_fem_data=fem_data_1)
        desired_indices = np.array([
            [0, 3, 2, 1, -1],
            [1, 0, 3, 2, -1],
            [2, 0, 3, 1, -1],
            [3, 0, 2, 1, -1],
        ])
        np.testing.assert_array_equal(indices, desired_indices)

        indices, vectors = \
            fem_data_1.nearest_neighbor_search_from_nodes_to_nodes(
                3, distance_upper_bound=1.2, target_fem_data=fem_data_1
            )
        desired_indices = np.array([
            [0, 3, 2],
            [1, 0, -1],
            [2, 0, -1],
            [3, 0, -1],
        ])
        np.testing.assert_array_equal(indices, desired_indices)

    def test_nearest_neighbor_search_from_elements_to_nodes(self):
        fem_data_1 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri', read_npy=False, save=False)
        indices, dists = \
            fem_data_1.nearest_neighbor_search_from_elements_to_nodes(4)
        desired = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, (1 / 3)**.5],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_almost_equal(dists, desired)

    def test_nearest_neighbor_search_from_nodes_to_elements(self):
        fem_data_1 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri', read_npy=False, save=False)
        indices, dists = \
            fem_data_1.nearest_neighbor_search_from_nodes_to_elements(4)
        desired = np.array([
            [0.0, 0.0, 0.0, (1 / 3)**.5],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_almost_equal(dists, desired)

    def test_nearest_neighbor_search_from_elements_to_elements(self):
        fem_data_1 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri', read_npy=False, save=False)
        indices, dists = \
            fem_data_1.nearest_neighbor_search_from_elements_to_elements(4)
        desired = np.array([
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ])
        np.testing.assert_almost_equal(dists, desired)

    def test_hausdorff_distance_nodes(self):
        fem_data_1 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri', read_npy=False, save=False)
        fem_data_2 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri_2', read_npy=False, save=False)

        actual = fem_data_1.calculate_hausdorff_distance_nodes(fem_data_2)
        desired = 2 ** .5
        np.testing.assert_almost_equal(actual, desired)

    def test_hausdorff_distance_elements(self):
        fem_data_1 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri', read_npy=False, save=False)
        fem_data_2 = FEMData.read_directory(
            'obj', 'tests/data/obj/tri_2', read_npy=False, save=False)

        actual = fem_data_1.calculate_hausdorff_distance_elements(fem_data_2)
        desired = (4 / 3) ** .5
        np.testing.assert_almost_equal(actual, desired, decimal=1)

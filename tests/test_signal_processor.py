import os
from pathlib import Path
import shutil
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from femio.fem_attribute import FEMAttribute
from femio.fem_data import FEMData
from femio.util import brick_generator


RUN_FISTR = True


FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'


class TestSignalProcessor(unittest.TestCase):

    def test_calculate_moving_average_elemental_data(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1', read_npy=False)
        adj = np.array([
            [1., .1, .1, .1, .1],
            [.1, 1., .1, .1, .1],
            [.1, .1, 1., .1, .1],
            [.1, .1, .1, 1., 0.],
            [.1, .1, .1, 0., 1.],
        ])
        elemental_data = np.array([
            [1e0, 2e0, 3e0],
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
        ])
        normalizers = (1 + np.array([
            .4, .4, .4, .3, .3])[:, None]) ** -1

        # Moving average with 1 hop
        desired_ave1 = adj @ elemental_data * normalizers
        actual_ave1 = fem_data.calculate_moving_average_elemental_data(
            elemental_data)
        np.testing.assert_almost_equal(actual_ave1, desired_ave1)

        # Moving average with 2 hops
        desired_ave2 = adj @ (adj @ elemental_data * normalizers) * normalizers
        actual_ave2 = fem_data.calculate_moving_average_elemental_data(
            elemental_data, hops=2)
        np.testing.assert_almost_equal(actual_ave2, desired_ave2)

    def test_calculate_moving_average_nodal_data(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1', read_npy=False)
        adj = np.array([
            [1., .1, .1, .1, 0., 0., .1, .1],
            [.1, 1., .1, .1, .1, 0., .1, .1],
            [.1, .1, 1., .1, .1, .1, 0., .1],
            [.1, .1, .1, 1., .1, .1, 0., 0.],
            [0., .1, .1, .1, 1., .1, 0., 0.],
            [0., 0., .1, .1, .1, 1., 0., 0.],
            [.1, .1, 0., 0., 0., 0., 1., .1],
            [.1, .1, .1, 0., 0., 0., .1, 1.],
        ])
        nodal_data = np.array([
            [1e0, 2e0, 3e0],
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
            [1e5, 2e5, 3e5],
            [1e6, 2e6, 3e6],
            [1e7, 2e7, 3e7],
        ])
        normalizers = (1 + np.array([
            .5, .6, .6, .5, .4, .3, .3, .4])[:, None]) ** -1

        # Moving average with 1 hop
        desired_ave1 = adj @ nodal_data * normalizers
        actual_ave1 = fem_data.calculate_moving_average_nodal_data(nodal_data)
        np.testing.assert_almost_equal(actual_ave1, desired_ave1)

        # Moving average with 2 hops
        desired_ave2 = adj @ (adj @ nodal_data * normalizers) * normalizers
        actual_ave2 = fem_data.calculate_moving_average_nodal_data(
            nodal_data, hops=2)
        np.testing.assert_almost_equal(actual_ave2, desired_ave2)

    def test_calculate_moving_average_tri(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tri', read_npy=False)
        adj = np.array([
            [1., .1, .1, 0.],
            [.1, 1., .1, .1],
            [.1, .1, 1., .1],
            [0., .1, .1, 1.],
        ])
        nodal_data = np.array([
            [1e0, 2e0, 3e0],
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
        ])
        normalizers = (1 + np.array([
            .2, .3, .3, .2])[:, None]) ** -1

        # Moving average with 1 hop
        desired_ave1 = adj @ nodal_data * normalizers
        actual_ave1 = fem_data.calculate_moving_average_nodal_data(nodal_data)
        np.testing.assert_almost_equal(actual_ave1, desired_ave1)

        # Moving average with 2 hops
        desired_ave2 = adj @ (adj @ nodal_data * normalizers) * normalizers
        actual_ave2 = fem_data.calculate_moving_average_nodal_data(
            nodal_data, hops=2)
        np.testing.assert_almost_equal(actual_ave2, desired_ave2)

    def test_calculate_diffusion_elemental_data(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1', read_npy=False)
        l_mat = np.array([
            [-4., 1., 1., 1., 1.],
            [1., -4., 1., 1., 1.],
            [1., 1., -4., 1., 1.],
            [1., 1., 1., -3., 0.],
            [1., 1., 1., 0., -3.],
        ])
        elemental_data = np.array([
            [1e0, 2e0, 3e0],
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
        ])

        # Diffuse with 1 hop
        desired_ave1 = elemental_data + .0001 * l_mat @ elemental_data
        actual_ave1 = fem_data.calculate_diffusion_elemental_data(
            elemental_data, weight=.0001)
        np.testing.assert_almost_equal(actual_ave1, desired_ave1)

        # Diffuse with 2 hops
        desired_ave2 = desired_ave1 + .0001 * l_mat @ desired_ave1
        actual_ave2 = fem_data.calculate_diffusion_elemental_data(
            elemental_data, hops=2, weight=.0001)
        np.testing.assert_almost_equal(actual_ave2, desired_ave2)

        # Confirm that summation is preserved
        np.testing.assert_almost_equal(
            np.sum(elemental_data, axis=0), np.sum(actual_ave2, axis=0))

    def test_calculate_moving_average_nodal_data_tet2(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet2', read_npy=False)
        adj = np.array([
            [1, .1, .1, .1, 0],
            [.1, 1, .1, .1, .1],
            [.1, .1, 1, .1, .1],
            [.1, .1, .1, 1, .1],
            [0, .1, .1, .1, 1],
        ])
        nodal_data = np.array([
            [1e0, 2e0, 3e0],
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
        ])
        normalizers = (1 + np.array([
            .3, .4, .4, .4, .3])[:, None]) ** -1

        # Moving average with 1 hop
        desired_ave1 = adj @ nodal_data * normalizers
        actual_ave1 = fem_data.calculate_moving_average_nodal_data(nodal_data)
        np.testing.assert_almost_equal(actual_ave1, desired_ave1)

        # Moving average with 2 hops
        desired_ave2 = adj @ (adj @ nodal_data * normalizers) * normalizers
        actual_ave2 = fem_data.calculate_moving_average_nodal_data(
            nodal_data, hops=2)
        np.testing.assert_almost_equal(actual_ave2, desired_ave2)

    def test_calculate_moving_average_nodal_data_disordered(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1_disordered', read_npy=False)
        adj = np.array([
            [1., .1, .1, .1, 0., 0., .1, .1],
            [.1, 1., .1, .1, .1, 0., .1, .1],
            [.1, .1, 1., .1, .1, .1, 0., .1],
            [.1, .1, .1, 1., .1, .1, 0., 0.],
            [0., .1, .1, .1, 1., .1, 0., 0.],
            [0., 0., .1, .1, .1, 1., 0., 0.],
            [.1, .1, 0., 0., 0., 0., 1., .1],
            [.1, .1, .1, 0., 0., 0., .1, 1.],
        ])
        nodal_data = np.array([
            [1e0, 2e0, 3e0],
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
            [1e5, 2e5, 3e5],
            [1e6, 2e6, 3e6],
            [1e7, 2e7, 3e7],
        ])
        normalizers = (1 + np.array([
            .5, .6, .6, .5, .4, .3, .3, .4])[:, None]) ** -1

        # Moving average with 1 hop
        desired_ave1 = adj @ nodal_data * normalizers
        actual_ave1 = fem_data.calculate_moving_average_nodal_data(nodal_data)
        np.testing.assert_almost_equal(actual_ave1, desired_ave1)

        # Moving average with 2 hops
        desired_ave2 = adj @ (adj @ nodal_data * normalizers) * normalizers
        actual_ave2 = fem_data.calculate_moving_average_nodal_data(
            nodal_data, hops=2)
        np.testing.assert_almost_equal(actual_ave2, desired_ave2)

    def test_calculate_moving_average_nodal_data_skipped(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1_skipped', read_npy=False)
        adj = np.array([
            [1., .1, .1, .1, 0., 0., .1, .1],
            [.1, 1., .1, .1, .1, 0., .1, .1],
            [.1, .1, 1., .1, .1, .1, 0., .1],
            [.1, .1, .1, 1., .1, .1, 0., 0.],
            [0., .1, .1, .1, 1., .1, 0., 0.],
            [0., 0., .1, .1, .1, 1., 0., 0.],
            [.1, .1, 0., 0., 0., 0., 1., .1],
            [.1, .1, .1, 0., 0., 0., .1, 1.],
        ])
        nodal_data = np.array([
            [1e0, 2e0, 3e0],
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
            [1e5, 2e5, 3e5],
            [1e6, 2e6, 3e6],
            [1e7, 2e7, 3e7],
        ])
        normalizers = (1 + np.array([
            .5, .6, .6, .5, .4, .3, .3, .4])[:, None]) ** -1

        # Moving average with 1 hop
        desired_ave1 = adj @ nodal_data * normalizers
        actual_ave1 = fem_data.calculate_moving_average_nodal_data(nodal_data)
        np.testing.assert_almost_equal(actual_ave1, desired_ave1)

        # Moving average with 2 hops
        desired_ave2 = adj @ (adj @ nodal_data * normalizers) * normalizers
        actual_ave2 = fem_data.calculate_moving_average_nodal_data(
            nodal_data, hops=2)
        np.testing.assert_almost_equal(actual_ave2, desired_ave2)

    def test_calculate_median_filter(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1', read_npy=False)
        nodal_data = np.array([
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
            [5e0, 2e0, 3e0],
            [1e0, 2e0, 7e0],
            [1e0, 7e0, 7e0],
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
        ])

        # Moving average with 1 hop
        desired_data = np.array([
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 5e0],
            [1e0, 2e0, 3e0],
            [1e0, 2e0, 3e0],
        ])
        actual_data = fem_data.calculate_median_filter(
            nodal_data, mode='nodal')
        np.testing.assert_almost_equal(actual_data, desired_data)

    def test_calculate_median_filter_element(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1', read_npy=False)
        elemental_data = np.array([
            [1e6, 2e3],
            [1e6, 2e5],
            [1e6, 2e3],
            [1e3, 2e3],
            [1e6, 2e3],
        ])

        desired_data = np.array([
            [1e6, 2e3],
            [1e6, 2e3],
            [1e6, 2e3],
            [1e6, 2e3],
            [1e6, 2e3],
        ])
        actual_data = fem_data.calculate_median_filter(
            elemental_data, mode='elemental')
        np.testing.assert_almost_equal(actual_data, desired_data)

    def test_calculate_median_filter_element_mixture(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid', read_npy=False)
        elemental_data = np.array([
            [1e6, 2e3],
            [1e6, 2e5],
            [1e6, 2e3],
            [1e3, 2e3],
        ])

        desired_data = np.array([
            [1e6, 2e3],
            [1e6, 2e3],
            [1e6, 2e3],
            [(1e6 + 1e3) / 2, 2e3],
        ])
        actual_data = fem_data.calculate_median_filter(
            elemental_data, mode='elemental')
        np.testing.assert_almost_equal(actual_data, desired_data)

    def test_nodal_spatial_gradient_tet2(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/tet2_cube',
            read_npy=False, read_mesh_only=True, save=False)
        xs = fem_data.nodal_data.get_attribute_data('node')
        actual_grads = fem_data.calculate_nodal_spatial_gradients(xs[:, [0]])

        np.testing.assert_almost_equal(
            np.mean(actual_grads, axis=0)[:, 0], [1., 0., 0.], decimal=2)
        if False:
            order1_fem_data = fem_data.to_first_order()
            mean_volumes = order1_fem_data.convert_elemental2nodal(
                order1_fem_data.calculate_element_volumes(), mode='mean')
            centered_volumes = order1_fem_data.convert_elemental2nodal(
                order1_fem_data.calculate_element_volumes(), mode='effective')
            order1_fem_data.nodal_data.update({
                'grad_x': FEMAttribute(
                    'grad_x', order1_fem_data.nodes.ids,
                    actual_grads[:, :, 0]),
                'mean_volume': FEMAttribute(
                    'mean_volume', order1_fem_data.nodes.ids,
                    mean_volumes),
                'centered_volume': FEMAttribute(
                    'centered_volume', order1_fem_data.nodes.ids,
                    centered_volumes),
            })
            order1_fem_data.write('ucd', 'temp.inp')

    def test_calculate_spatial_gradient_adjacency_matrix_element(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_cross',
            read_npy=False, read_mesh_only=True, save=False)
        actual_adjs = fem_data.calculate_spatial_gradient_adjacency_matrices(
            'elemental', n_hop=1)

        expected_adj0 = np.array(
            [[0., 0.05, 0., 0., 0., -0.05, 0.],
             [-0.05, 0.3, -0.05, -0.1, -0.05, 0., -0.05],
             [0., 0.05, 0., 0., 0., -0.05, 0.],
             [0., .5/6, 0., 0., 0., -.5/6, 0.],
             [0., 0.05, 0., 0., 0., -0.05, 0.],
             [0.05, 0., 0.05, 0.1, 0.05, -0.3, 0.05],
             [0., 0.05, 0., 0., 0., -0.05, 0.]]
        ) * 3.
        expected_adj1 = np.array(
            [[0., 0., -0.05, 0., 0.05, 0., 0.],
             [0., 0., -0.05, 0., 0.05, 0., 0.],
             [0.05, 0.05, -0.3, 0.1, 0., 0.05, 0.05],
             [0., 0., -.5/6, 0., .5/6, 0., 0.],
             [-0.05, -0.05, 0., -0.1, 0.3, -0.05, -0.05],
             [0., 0., -0.05, 0., 0.05, 0., 0.],
             [0., 0., -0.05, 0., 0.05, 0., 0.]]
        ) * 3.
        expected_adj2 = np.array(
            [[-0.3, 0.05, 0.05, 0.1, 0.05, 0.05, 0.],
             [-0.05, 0., 0., 0., 0., 0., 0.05],
             [-0.05, 0., 0., 0., 0., 0., 0.05],
             [-.5/6, 0., 0., 0., 0., 0., .5/6],
             [-0.05, 0., 0., 0., 0., 0., 0.05],
             [-0.05, 0., 0., 0., 0., 0., 0.05],
             [0., -0.05, -0.05, -0.1, -0.05, -0.05, 0.3]]
        ) * 3.
        np.testing.assert_almost_equal(
            actual_adjs[0].toarray(), expected_adj0)
        np.testing.assert_almost_equal(
            actual_adjs[1].toarray(), expected_adj1)
        np.testing.assert_almost_equal(
            actual_adjs[2].toarray(), expected_adj2)

        np.testing.assert_array_equal(actual_adjs[0].shape, (7, 7))
        np.testing.assert_array_equal(actual_adjs[1].shape, (7, 7))
        np.testing.assert_array_equal(actual_adjs[2].shape, (7, 7))

    def test_calculate_data_diff_adjs(self):
        fem_data = FEMData()
        adj = sp.coo_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))
        data = np.array([[10., -10.], [10., 0.], [20., 0.]])
        actual_adjs = fem_data.calculate_data_diff_adjs(adj, data)

        np.testing.assert_almost_equal(
            actual_adjs[0].toarray(), np.array([
                [0., 0., 0.], [0., 0., 10.], [0., -10., 0.]]))
        np.testing.assert_almost_equal(
            actual_adjs[1].toarray(), np.array([
                [0., 10., 0.], [-10., 0., 0.], [0., 0., 0.]]))

    def test_calculate_spatial_gradient_adjacency_matrix_node(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_cross',
            read_npy=False, read_mesh_only=True, save=False)
        actual_adjs = fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=1)
        # expected_adj00 = np.array([
        #     0.166666667, 0., -0.071428571, -0.035714286, 0., 0., 0., 0., 0.,
        #     0., 0., -0.035714286, -0.023809524, 0., 0., 0., 0., 0., 0., 0.,
        #     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) * 3.
        # np.testing.assert_almost_equal(
        #     actual_adjs[0].toarray()[0], expected_adj00)

        np.testing.assert_array_equal(actual_adjs[0].shape, (32, 32))
        np.testing.assert_array_equal(actual_adjs[1].shape, (32, 32))
        np.testing.assert_array_equal(actual_adjs[2].shape, (32, 32))

    def test_nodal_spatial_gradient_large_data(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/middle_hex',
            read_npy=False, read_mesh_only=True, save=False)
        center_node_index = 666

        xs = fem_data.nodal_data.get_attribute_data('NODE')
        fem_data.nodal_data.update({
            'spatial_nodal_data':
            FEMAttribute(
                'spatial_nodal_data',
                ids=fem_data.nodes.ids,
                data=np.stack(
                    [2. * xs[:, 0], 3. * xs[:, 1] + 4. * xs[:, 2]**2], axis=1)
            )
        })
        actual_grads = fem_data.calculate_nodal_spatial_gradients(
            fem_data.nodal_data.get_attribute_data('spatial_nodal_data'),
            n_hop=3)
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 0],
            [2., 0., 0.])
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 1],
            [0., 3., 8. * fem_data.nodes.data[center_node_index, 2]])

    def test_nodal_spatial_gradient_large_data_with_kernel(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/middle_hex',
            read_npy=False, read_mesh_only=True, save=False)
        center_node_index = 666

        grad_id_adjs = fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=3)
        grad_exp_adjs = fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=3, kernel='exp', alpha=3.)
        grad_id_adjs_data = grad_id_adjs[0].tocsr()[0].data
        grad_exp_adjs_data = grad_exp_adjs[0].tocsr()[0].data
        np.testing.assert_array_less(
            grad_exp_adjs_data[np.abs(grad_id_adjs_data) < 5e-3],
            grad_id_adjs_data[np.abs(grad_id_adjs_data) < 5e-3])

        # # Plot data
        # grad_gauss_adjs = \
        #     fem_data.calculate_spatial_gradient_adjacency_matrices(
        #         'nodal', n_hop=3, kernel='gauss', alpha=3.)
        # grad_chisquare_adjs = \
        #     fem_data.calculate_spatial_gradient_adjacency_matrices(
        #         'nodal', n_hop=3, kernel='chisquare', k=10)
        # import matplotlib.pyplot as plt
        # plt.plot(
        #     grad_id_adjs[0].tocsr()[center_node_index].data,
        #     grad_exp_adjs[0].tocsr()[center_node_index].data,
        #     '.', label='exp')
        # plt.plot(
        #     grad_id_adjs[0].tocsr()[center_node_index].data,
        #     grad_gauss_adjs[0].tocsr()[center_node_index].data,
        #     '.', label='gauss')
        # plt.plot(
        #     grad_id_adjs[0].tocsr()[center_node_index].data,
        #     grad_chisquare_adjs[0].tocsr()[center_node_index].data,
        #     '.', label='chisquare')
        # plt.legend()
        # plt.show()

        xs = fem_data.nodal_data.get_attribute_data('NODE')
        fem_data.nodal_data.update({
            'spatial_nodal_data':
            FEMAttribute(
                'spatial_nodal_data',
                ids=fem_data.nodes.ids,
                data=np.stack(
                    [2. * xs[:, 0], 3. * xs[:, 1] + 4. * xs[:, 2]**2], axis=1)
            )
        })
        actual_grads = fem_data.calculate_nodal_spatial_gradients(
            fem_data.nodal_data.get_attribute_data('spatial_nodal_data'),
            n_hop=3, kernel='exp', alpha=.1)
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 0],
            [2., 0., 0.])
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 1],
            [0., 3., 8. * fem_data.nodes.data[center_node_index, 2]])

        actual_grads = fem_data.calculate_nodal_spatial_gradients(
            fem_data.nodal_data.get_attribute_data('spatial_nodal_data'),
            n_hop=3, kernel='gauss', alpha=2.)
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 0],
            [2., 0., 0.])
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 1],
            [0., 3., 8. * fem_data.nodes.data[center_node_index, 2]])

        actual_grads = fem_data.calculate_nodal_spatial_gradients(
            fem_data.nodal_data.get_attribute_data('spatial_nodal_data'),
            n_hop=3, kernel='chisquare', k=10)
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 0],
            [2., 0., 0.])
        np.testing.assert_almost_equal(
            actual_grads[center_node_index, :, 1],
            [0., 3., 8. * fem_data.nodes.data[center_node_index, 2]])

    def test_elemental_spatial_gradient(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_cross',
            read_npy=False, read_mesh_only=True, save=False)
        xs = fem_data.convert_nodal2elemental('NODE', calc_average=True)
        fem_data.elemental_data.update_data(
            fem_data.elements.ids, {'spatial_elemental_data': np.stack(
                [2. * xs[:, 0], 3. * xs[:, 1] + 4. * xs[:, 2]**2], axis=1)}
        )
        actual_grads = fem_data.calculate_elemental_spatial_gradients(
            fem_data.elemental_data.get_attribute_data(
                'spatial_elemental_data'))
        expected_grads0 = np.array([
            [3. * 2. * 2. * (2. / 8. * 2 + 2. / 4. * 0) / 5, 0., 0.],
            [3. * 2. * 2. * (2. / 8. * 4 + 2. / 4. * 1) / 5, 0., 0.],
            [3. * 2. * 2. * (2. / 8. * 2 + 2. / 4. * 0) / 5, 0., 0.],
            [3. * 2. * 2. * (2. / 8. * 0 + 2. / 4. * 2) / 6, 0., 0.],
            [3. * 2. * 2. * (2. / 8. * 2 + 2. / 4. * 0) / 5, 0., 0.],
            [3. * 2. * 2. * (2. / 8. * 4 + 2. / 4. * 1) / 5, 0., 0.],
            [3. * 2. * 2. * (2. / 8. * 2 + 2. / 4. * 0) / 5, 0., 0.],
        ])
        # expected_grads1 = np.stack([
        #     np.zeros(7), np.ones(7)*3., 8. * xs[:, 2]], axis=1)
        np.testing.assert_almost_equal(
            actual_grads[:, :, 0], expected_grads0)
        # np.testing.assert_almost_equal(
        #     actual_grads[:, :, 1], expected_grads1)
        np.testing.assert_array_equal(
            actual_grads.shape, (7, 3, 2))

    def test_elemental_spatial_gradient_large_hops(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_cross',
            read_npy=False, read_mesh_only=True, save=False)
        xs = fem_data.convert_nodal2elemental('NODE', calc_average=True)
        fem_data.elemental_data.update_data(
            fem_data.elements.ids, {'spatial_elemental_data': np.stack(
                [2. * xs[:, 0], 3. * xs[:, 1] + 4. * xs[:, 2]**2], axis=1)})
        actual_grads = fem_data.calculate_elemental_spatial_gradients(
            fem_data.elemental_data.get_attribute_data(
                'spatial_elemental_data'), n_hop=10)
        expected_grads0 = np.array([
            [3. * (1. / 3. * 1. + 0.), 0., 0.],
            [3. * (1. / 3. * 1. + 1.), 0., 0.],
            [3. * (1. / 3. * 1. + 0.), 0., 0.],
            [3. * (1. / 3. * 2. + 0.), 0., 0.],
            [3. * (1. / 3. * 1. + 0.), 0., 0.],
            [3. * (1. / 3. * 1. + 1.), 0., 0.],
            [3. * (1. / 3. * 1. + 0.), 0., 0.],
        ])
        np.testing.assert_almost_equal(
            actual_grads[:, :, 0], expected_grads0)
        np.testing.assert_array_equal(
            actual_grads.shape, (7, 3, 2))

    def test_elemental_spatial_gradient_large_data(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/middle_hex',
            read_npy=False, read_mesh_only=True, save=False)
        center_element_index = 445

        xs = fem_data.convert_nodal2elemental('NODE', calc_average=True)
        fem_data.elemental_data.update_data(
            fem_data.elements.ids, {'spatial_elemental_data': np.stack(
                [2. * xs[:, 0], 3. * xs[:, 1] + 4. * xs[:, 2]**2], axis=1)})
        actual_grads = fem_data.calculate_elemental_spatial_gradients(
            fem_data.elemental_data.get_attribute_data(
                'spatial_elemental_data'), n_hop=3)
        element_position = fem_data.convert_nodal2elemental(
            'NODE', calc_average=True)
        np.testing.assert_almost_equal(
            actual_grads[center_element_index, :, 0],
            [2., 0., 0.])
        np.testing.assert_almost_equal(
            actual_grads[center_element_index, :, 1],
            [0., 3., 2. * 4. * element_position[center_element_index, 2]])

    def test_elemental_spatial_gradient_large_data_with_kernel(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/middle_hex',
            read_npy=False, read_mesh_only=True, save=False)
        center_element_index = 445

        xs = fem_data.convert_nodal2elemental('NODE', calc_average=True)
        fem_data.elemental_data.update_data(
            fem_data.elements.ids, {'spatial_elemental_data': np.stack(
                [2. * xs[:, 0], 3. * xs[:, 1] + 4. * xs[:, 2]**2], axis=1)})
        actual_grads = fem_data.calculate_elemental_spatial_gradients(
            fem_data.elemental_data.get_attribute_data(
                'spatial_elemental_data'), n_hop=3,
            kernel='exp', alpha=2.)
        element_position = fem_data.convert_nodal2elemental(
            'NODE', calc_average=True)
        np.testing.assert_almost_equal(
            actual_grads[center_element_index, :, 0],
            [2., 0., 0.])
        np.testing.assert_almost_equal(
            actual_grads[center_element_index, :, 1],
            [0., 3., 2. * 4. * element_position[center_element_index, 2]])

    def test_nodal_spatial_gradient(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_cross',
            read_npy=False, read_mesh_only=True, save=False)
        xs = fem_data.nodal_data.get_attribute_data('node')
        actual_grads = fem_data.calculate_nodal_spatial_gradients(xs)

        np.testing.assert_almost_equal(
            np.mean(actual_grads[:, :, 0], axis=0), [1., 0., 0.])
        np.testing.assert_almost_equal(
            np.mean(actual_grads[:, :, 1], axis=0), [0., 1., 0.])
        np.testing.assert_almost_equal(
            np.mean(actual_grads[:, :, 2], axis=0), [0., 0., 1.])

    def test_integrate(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_3', read_npy=False, save=False)
        t_init = fem_data.nodal_data.get_attribute_data('t_init')
        inputs = np.concatenate([t_init, t_init * 2], axis=-1)
        integrated_elements = fem_data.integrate_elements(inputs)
        integrated = fem_data.integrate(inputs)

        desired_integrated_t_init = np.array([
            8 * (40 + 50 + 60 + 70),
            32 * (10 + 20 + 30 + 70),
            8 * (10 + 50 + 60 + 70),
            16 * (10 + 20 + 60 + 70),
        ]) / (2 * 3) / 4
        desired_integrated_elements = np.stack([
            desired_integrated_t_init, desired_integrated_t_init * 2], axis=-1)
        np.testing.assert_almost_equal(
            integrated_elements, desired_integrated_elements)
        np.testing.assert_almost_equal(
            integrated, np.sum(desired_integrated_elements, axis=0))

    def test_fistr_convert_lte_1(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_full_easy', read_npy=False,
            save=False)
        desired_lte_full = fem_data.elemental_data.get_attribute_data(
            'lte_full')

        # Global -> local
        fem_data.convert_lte_global2local()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('lte')[0] * 1e7,
            np.array([1.0, 2.0, 3.0]))

        # Local -> global
        fem_data.elemental_data.pop(
            'linear_thermal_expansion_coefficient_full', None)
        fem_data.convert_lte_local2global()
        actual_lte_full = fem_data.elemental_data.get_attribute_data(
            'lte_full')
        np.testing.assert_almost_equal(
            actual_lte_full, desired_lte_full)

    def test_fistr_convert_lte_full_to_lte_local_1(self):
        data_dir = Path('tests/data/fistr/thermal_full')
        fem_data = FEMData.read_directory(
            'fistr', data_dir, read_npy=False, save=False)
        desired_lte_full = fem_data.elemental_data.get_attribute_data(
            'lte_full')

        # Global -> local
        fem_data.convert_lte_global2local()
        fem_data.elemental_data.pop(
            'linear_thermal_expansion_coefficient_full', None)
        write_local = data_dir / 'to_local'
        if write_local.exists():
            shutil.rmtree(write_local)
        fem_data.write('fistr', write_local / 'mesh')
        if RUN_FISTR:
            os.system(f"cd {write_local} "
                      + "&& fistr1 > /dev/null 2>&1")
            fem_data_local = FEMData.read_directory(
                'fistr', write_local, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                fem_data_local.nodal_data.get_attribute_data('DISPLACEMENT'),
                fem_data.nodal_data.get_attribute_data('DISPLACEMENT'))

        # Local -> global
        fem_data.convert_lte_local2global()
        fem_data.elemental_data.pop(
            'linear_thermal_expansion_coefficient', None)
        fem_data.elemental_data.pop(
            'ORIENTATION', None)
        actual_lte_full = fem_data.elemental_data.get_attribute_data(
            'lte_full')
        np.testing.assert_almost_equal(
            actual_lte_full, desired_lte_full)
        write_global = data_dir / 'to_global'
        if write_global.exists():
            shutil.rmtree(write_global)
        fem_data.write('fistr', write_global / 'mesh')
        if RUN_FISTR:
            os.system(f"cd {write_global} "
                      + "&& fistr1 > /dev/null 2>&1")
            fem_data_global = FEMData.read_directory(
                'fistr', write_global, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                fem_data_global.nodal_data.get_attribute_data('DISPLACEMENT'),
                fem_data.nodal_data.get_attribute_data('DISPLACEMENT'))

    def test_convert_nodal2elemental_node(self):
        """Node positions should be correctly collected element by element."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        nodes = fem_data.nodes.data
        desired_nodes = np.array([
            nodes[:8, :],
            nodes[4:, :],
        ])
        actual_nodes = fem_data.convert_nodal2elemental(
            'node')
        np.testing.assert_almost_equal(
            desired_nodes, actual_nodes)

    def test_convert_nodal2elemental_displacement(self):
        """Elemental data should be calculated correctly from nodal data."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        displacements = fem_data.nodal_data['DISPLACEMENT'].data
        desired_elemental_displacements = np.array([
            displacements[:8, :],
            displacements[4:, :],
        ])
        actual_elemental_displacements = fem_data.convert_nodal2elemental(
            'DISPLACEMENT')
        np.testing.assert_almost_equal(
            desired_elemental_displacements, actual_elemental_displacements)

    def test_convert_nodal2elemental_mean_displacement(self):
        """Elemental data should be calculated correctly from nodal data."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        displacements = fem_data.nodal_data['DISPLACEMENT'].data
        desired_elemental_displacements = np.array([
            np.mean(displacements[:8, :], axis=0),
            np.mean(displacements[4:, :], axis=0),
        ])
        actual_elemental_displacements = fem_data.convert_nodal2elemental(
            'DISPLACEMENT', calc_average=True)
        np.testing.assert_almost_equal(
            desired_elemental_displacements, actual_elemental_displacements)

    def test_convert_nodal2elemental_direct_input(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        displacements = fem_data.nodal_data['DISPLACEMENT'].data
        actual_elemental_displacements = fem_data.convert_nodal2elemental(
            displacements, calc_average=True)
        desired_elemental_displacements = \
            fem_data.convert_nodal2elemental(
                'DISPLACEMENT', calc_average=True)
        np.testing.assert_almost_equal(
            actual_elemental_displacements, desired_elemental_displacements)

    def test_convert_nodal2elemental_temeprature(self):
        """Elemental data should be calculated correctly from nodal data."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        temperatures = fem_data.nodal_data['INITIAL_TEMPERATURE'].data
        desired_elemental_temperature = np.array([
            np.ravel(temperatures[:8, :]),
            np.ravel(temperatures[4:, :]),
        ])
        actual_elemental_temperature = fem_data.convert_nodal2elemental(
            'INITIAL_TEMPERATURE', ravel=True)
        np.testing.assert_almost_equal(
            desired_elemental_temperature, actual_elemental_temperature)

    def test_convert_nodal2elemental_mean_temeprature(self):
        """Elemental data should be calculated correctly from nodal data."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        temperatures = fem_data.nodal_data['INITIAL_TEMPERATURE'].data
        desired_elemental_temperature = np.array([
            np.mean(temperatures[:8, :], axis=0),
            np.mean(temperatures[4:, :], axis=0),
        ])
        actual_elemental_temperature = fem_data.convert_nodal2elemental(
            'INITIAL_TEMPERATURE', calc_average=True)
        np.testing.assert_almost_equal(
            desired_elemental_temperature, actual_elemental_temperature)

    def test_convert_elemental2nodal_mean(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_e2n', read_npy=False,
            save=False)
        elemental_data = np.array([
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
        ])
        actual_converted_data = fem_data.convert_elemental2nodal(
            elemental_data, mode='mean')
        desired_converted_data = np.stack([
            (elemental_data[0] * 1/6 + elemental_data[1] * 1/3) / (1/6 + 1/3),
            elemental_data[0],
            (elemental_data[0] * 1/6 + elemental_data[1] * 1/3
             + elemental_data[2] * 1) / (1/6 + 1/3 + 1),
            (elemental_data[0] * 1/6 + elemental_data[1] * 1/3
             + elemental_data[2] * 1) / (1/6 + 1/3 + 1),
            elemental_data[2],
            (elemental_data[1] * 1/3 + elemental_data[2] * 1) / (1/3 + 1),
        ])
        np.testing.assert_almost_equal(
            actual_converted_data, desired_converted_data)

    def test_convert_elemental2nodal_polygon(self):
        fem_data = FEMData.read_directory(
            'vtp', 'tests/data/vtp/polys', read_npy=False, save=False)
        elemental_data = np.array([
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
        ])
        actual_converted_mean_data = fem_data.convert_elemental2nodal(
            elemental_data, mode='mean')
        v0 = 1.25
        v2 = 1.16
        d = elemental_data
        desired_converted_mean_data = np.array([
            (d[2] * v2 + d[0] * v0) / (v2 + v0),
            d[0],
            d[0],
            (d[0] + d[3]) / 2,
            (d[0] * v0 + d[2] * v2 + d[3] * v0) / (v0 * 2 + v2),
            (d[2] * v2 + d[1] * v0) / (v0 + v2),
            d[1],
            d[1],
            (d[1] + d[3]) / 2,
            (d[1] * v0 + d[3] * v0 + d[2] * v2) / (v0 * 2 + v2),
            d[2],
            d[2],
            d[3],
        ])
        np.testing.assert_almost_equal(
            actual_converted_mean_data, desired_converted_mean_data,
            decimal=3)
        actual_converted_effective_data = fem_data.convert_elemental2nodal(
            elemental_data, mode='effective')
        v0 = 1 / 5
        v2 = 1 / 6
        desired_converted_effective_data = np.array([
            d[2] * v2 + d[0] * v0,
            d[0] * v0,
            d[0] * v0,
            d[0] * v0 + d[3] * v0,
            d[0] * v0 + d[2] * v2 + d[3] * v0,
            d[2] * v2 + d[1] * v0,
            d[1] * v0,
            d[1] * v0,
            d[1] * v0 + d[3] * v0,
            d[1] * v0 + d[3] * v0 + d[2] * v2,
            d[2] * v2,
            d[2] * v2,
            d[3] * v0,
        ])
        np.testing.assert_almost_equal(
            actual_converted_effective_data, desired_converted_effective_data)

    def test_convert_nodal2elemental_polygon(self):
        fem_data = FEMData.read_directory(
            'vtp', 'tests/data/vtp/polys', read_npy=False, save=False)
        d = fem_data.nodes.data
        actual_converted_mean_data = fem_data.convert_nodal2elemental(
            d, calc_average=True)
        desired_converted_mean_data = np.array([
            (d[0] + d[1] + d[2] + d[3] + d[4]) / 5,
            (d[5] + d[6] + d[7] + d[8] + d[9]) / 5,
            (d[0] + d[4] + d[9] + d[5] + d[11] + d[10]) / 6,
            (d[4] + d[3] + d[12] + d[8] + d[9]) / 5,
        ])
        np.testing.assert_almost_equal(
            actual_converted_mean_data, desired_converted_mean_data,
            decimal=3)

    def test_convert_elemental2nodal_mean_tet2_tet1(self):
        fem_data_tet1 = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_3', read_npy=False,
            save=False)
        fem_data_tet2 = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3', read_npy=False,
            save=False)
        elemental_data = np.array([
            [1e1, 2e1, 3e1],
            [1e2, 2e2, 3e2],
            [1e3, 2e3, 3e3],
            [1e4, 2e4, 3e4],
        ])
        converted_data_tet1 = fem_data_tet1.convert_elemental2nodal(
            elemental_data, mode='mean')
        converted_data_tet2 = fem_data_tet2.convert_elemental2nodal(
            elemental_data, mode='mean')
        np.testing.assert_almost_equal(
            converted_data_tet2[:7, :], converted_data_tet1)

    def test_convert_elemental2nodal_effective_volume(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_3', read_npy=False,
            save=False)
        volumes = fem_data.calculate_element_volumes()
        converted_volumes = fem_data.convert_elemental2nodal(
            volumes, mode='effective')
        desired_converted_volumes = np.array([
            np.sum(volumes[[1, 2, 3]]),
            np.sum(volumes[[1, 3]]),
            np.sum(volumes[[1]]),
            np.sum(volumes[[0]]),
            np.sum(volumes[[0, 2]]),
            np.sum(volumes[[0, 2, 3]]),
            np.sum(volumes[[0, 1, 2, 3]]),
        ]) / 4
        np.testing.assert_almost_equal(
            converted_volumes[:, 0], desired_converted_volumes)
        np.testing.assert_almost_equal(
            np.sum(converted_volumes), np.sum(volumes))

    def test_convert_elemental2nodal_effective_volume_tet2(self):
        tet_fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_3', read_npy=False,
            save=False)
        tet2_fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3', read_npy=False,
            save=False)
        volumes = tet_fem_data.calculate_element_volumes()
        tet_converted_volumes = tet_fem_data.convert_elemental2nodal(
            volumes, mode='effective')
        tet2_converted_volumes = tet2_fem_data.convert_elemental2nodal(
            volumes, mode='effective')

        np.testing.assert_almost_equal(
            tet2_converted_volumes, tet_converted_volumes)
        np.testing.assert_almost_equal(
            np.sum(tet2_converted_volumes), np.sum(volumes))

    def test_array2symmetric_matrix(self):
        in_array = np.array([
            [1.11, 1.22, 1.33, 1.12, 1.23, 1.31],
            [2.11, 2.22, 2.33, 2.12, 2.23, 2.31],
            [3.11, 3.22, 3.33, 3.12, 3.23, 3.31],
            [4.11, 4.22, 4.33, 4.12, 4.23, 4.31],
        ])
        desired_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        out_array = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False).convert_array2symmetric_matrix(in_array)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_array2symmetric_matrix_engineering(self):
        in_array = np.array([
            [1.11, 1.22, 1.33, 1.12 * 2, 1.23 * 2, 1.31 * 2],
            [2.11, 2.22, 2.33, 2.12 * 2, 2.23 * 2, 2.31 * 2],
            [3.11, 3.22, 3.33, 3.12 * 2, 3.23 * 2, 3.31 * 2],
            [4.11, 4.22, 4.33, 4.12 * 2, 4.23 * 2, 4.31 * 2],
        ])
        desired_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        out_array = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False).convert_array2symmetric_matrix(
                in_array, from_engineering=True)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_array2symmetric_matrix_order(self):
        in_array = np.array([
            [1.11, 1.12, 1.31, 1.22, 1.23, 1.33],
            [2.11, 2.12, 2.31, 2.22, 2.23, 2.33],
            [3.11, 3.12, 3.31, 3.22, 3.23, 3.33],
            [4.11, 4.12, 4.31, 4.22, 4.23, 4.33],
        ])
        desired_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        out_array = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False).convert_array2symmetric_matrix(
                in_array, order=[0, 3, 5, 1, 4, 2])
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_symmetric_matrix2array(self):
        in_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        desired_array = np.array([
            [1.11, 1.22, 1.33, 1.12, 1.23, 1.31],
            [2.11, 2.22, 2.33, 2.12, 2.23, 2.31],
            [3.11, 3.22, 3.33, 3.12, 3.23, 3.31],
            [4.11, 4.22, 4.33, 4.12, 4.23, 4.31],
        ])
        out_array = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False).convert_symmetric_matrix2array(in_array)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_symmetric_matrix2array_to_engineering(self):
        in_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        desired_array = np.array([
            [1.11, 1.22, 1.33, 1.12 * 2, 1.23 * 2, 1.31 * 2],
            [2.11, 2.22, 2.33, 2.12 * 2, 2.23 * 2, 2.31 * 2],
            [3.11, 3.22, 3.33, 3.12 * 2, 3.23 * 2, 3.31 * 2],
            [4.11, 4.22, 4.33, 4.12 * 2, 4.23 * 2, 4.31 * 2],
        ])
        out_array = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False).convert_symmetric_matrix2array(
                in_array, to_engineering=True)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_symmetric_matrix2array_order(self):
        in_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        desired_array = np.array([
            [1.11, 1.12, 1.31, 1.22, 1.23, 1.33],
            [2.11, 2.12, 2.31, 2.22, 2.23, 2.33],
            [3.11, 3.12, 3.31, 3.22, 3.23, 3.33],
            [4.11, 4.12, 4.31, 4.22, 4.23, 4.33],
        ])
        out_array = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False).convert_symmetric_matrix2array(
                in_array, order=[0, 3, 5, 1, 4, 2])
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_calculate_principal_components(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/lte_direction',
            read_npy=False, save=False)
        lte, directions, lte_vec = fem_data.calculate_principal_components(
            fem_data.elemental_data.get_attribute_data('lte_full'),
            from_engineering=True)

        np.testing.assert_almost_equal(
            lte[0, :],
            np.array([.01, .001, .0001]))
        np.testing.assert_almost_equal(
            directions[0],
            np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.]))
        np.testing.assert_almost_equal(
            lte_vec[0],
            np.array([.01, 0., 0., 0., .001, 0., 0., 0., .0001]))

        np.testing.assert_almost_equal(
            lte[1, :],
            np.array([0.01, 0.001, 0.0001]))
        np.testing.assert_almost_equal(
            directions[1],
            np.array([0., 0., 1., 0., 1., 0., -1., 0., 0.]))
        # raise ValueError(lte[1, :], directions[1], lte_vec[1])
        np.testing.assert_almost_equal(
            lte_vec[1],
            np.array([0., 0., .01, 0., .001, 0., -.0001, 0., 0.]))

        np.testing.assert_almost_equal(
            lte[2, :],
            np.array([0.003, 0.002, 0.001]))
        np.testing.assert_almost_equal(
            directions[2],
            np.array([
                0., 0., 1.,
                -1 / 2**.5, 1 / 2**.5, 0.,
                -1 / 2**.5, -1 / 2**.5, 0.]))
        np.testing.assert_almost_equal(
            lte_vec[2],
            np.array([
                0., 0., .003,
                -.002 / 2**.5, .002 / 2**.5, 0.,
                -.001 / 2**.5, -.001 / 2**.5, 0.]))

    def test_add_principal_vectors(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/lte_direction',
            read_npy=False, save=False)
        _, _, lte_vec = fem_data.calculate_principal_components(
            fem_data.elemental_data.get_attribute_data('lte_full'),
            from_engineering=True)
        fem_data.add_principal_vectors('lte_full')
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('principal_lte_full_1'),
            lte_vec[:, :3])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('principal_lte_full_2'),
            lte_vec[:, 3:6])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('principal_lte_full_3'),
            lte_vec[:, 6:])

    def test_invert_strain_xx(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/invert_strain_xx',
            read_npy=False, save=False)

        write_dir_name = Path('tests/data/fistr/write_invert_strain_xx')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)

        fem_data.elemental_data.pop(
            'linear_thermal_expansion_coefficient', None)
        fem_data.elemental_data.pop(
            'ORIENTATION', None)
        fem_data.elemental_data.pop(
            'INITIAL_TEMPERATURE', None)
        fem_data.elemental_data.pop(
            'CNT_TEMPERATURE', None)
        original_nodes = fem_data.nodes.data
        fem_data.nodes.data = fem_data.nodes.data \
            + fem_data.nodal_data.get_attribute_data('DISPLACEMENT')
        fem_data.settings['solution_type'] = 'EPS2DISP'
        fem_data.write('fistr', write_dir_name / 'mesh')

        istrain_file_name = write_dir_name / 'istrain.dat'
        with open(istrain_file_name, 'w') as f:
            f.write(f"342 {len(fem_data.elements.ids)}\n")
        pd.DataFrame(np.concatenate([
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN1'),
                is_engineering=True),
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN2'),
                is_engineering=True),
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN3'),
                is_engineering=True),
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN4'),
                is_engineering=True),
        ], axis=1), index=fem_data.elements.ids).to_csv(
            istrain_file_name, sep=' ', index=True, header=False,
            mode='a')

        if RUN_FISTR:
            os.system(f"cd {write_dir_name} "
                      + "&& fistr1 > /dev/null 2>&1")
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodes.data
                + written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                original_nodes)

    def test_invert_strain(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/invert_strain',
            read_npy=False, save=False)

        write_dir_name = Path('tests/data/fistr/write_invert_strain')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)

        fem_data.elemental_data.pop(
            'linear_thermal_expansion_coefficient', None)
        fem_data.elemental_data.pop(
            'ORIENTATION', None)
        fem_data.elemental_data.pop(
            'INITIAL_TEMPERATURE', None)
        fem_data.elemental_data.pop(
            'CNT_TEMPERATURE', None)
        original_nodes = fem_data.nodes.data
        fem_data.nodes.data = fem_data.nodes.data \
            + fem_data.nodal_data.get_attribute_data('DISPLACEMENT')
        fem_data.settings['solution_type'] = 'EPS2DISP'
        fem_data.write('fistr', write_dir_name / 'mesh')

        istrain_file_name = write_dir_name / 'istrain.dat'
        with open(istrain_file_name, 'w') as f:
            f.write(f"342 {len(fem_data.elements.ids)}\n")
        pd.DataFrame(np.concatenate([
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN1'),
                is_engineering=True),
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN2'),
                is_engineering=True),
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN3'),
                is_engineering=True),
            fem_data.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN4'),
                is_engineering=True),
        ], axis=1), index=fem_data.elements.ids).to_csv(
            istrain_file_name, sep=' ', index=True, header=False,
            mode='a')

        if RUN_FISTR:
            os.system(f"cd {write_dir_name} "
                      + "&& fistr1 > /dev/null 2>&1")
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodes.data
                + written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                original_nodes, decimal=3)

    def test_calculate_symmetric_matrices_from_eigens(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/lte_direction',
            read_npy=False, save=False)
        lte, directions, lte_vec = fem_data.calculate_principal_components(
            fem_data.elemental_data.get_attribute_data('lte_full'),
            from_engineering=True)

        calculated_lte_full = fem_data.calculate_array_from_eigens(
            lte, directions, to_engineering=True)
        np.testing.assert_almost_equal(
            calculated_lte_full,
            fem_data.elemental_data.get_attribute_data('lte_full'))

    def test_convert_vertices_to_edges(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1',
            read_npy=False, read_mesh_only=True, save=False)
        xs = fem_data.convert_nodal2elemental('NODE', calc_average=True)
        expected_differences = np.array([
            [xs[0], xs[1]],
            [xs[0], xs[2]],
            [xs[0], xs[3]],
            [xs[0], xs[4]],
            [xs[1], xs[0]],
            [xs[1], xs[2]],
            [xs[1], xs[3]],
            [xs[1], xs[4]],
            [xs[2], xs[0]],
            [xs[2], xs[1]],
            [xs[2], xs[3]],
            [xs[2], xs[4]],
            [xs[3], xs[0]],
            [xs[3], xs[1]],
            [xs[3], xs[2]],
            [xs[4], xs[0]],
            [xs[4], xs[1]],
            [xs[4], xs[2]],
        ])
        np.testing.assert_almost_equal(
            fem_data.convert_vertices_to_edges(
                xs, mode='elemental', include_self_loop=False),
            expected_differences)

    def test_calculate_elemental_edge_differences(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1',
            read_npy=False, read_mesh_only=True, save=False)
        xs = fem_data.convert_nodal2elemental('NODE', calc_average=True)
        expected_differences = np.array([
            xs[1] - xs[0],
            xs[2] - xs[0],
            xs[3] - xs[0],
            xs[4] - xs[0],
            xs[0] - xs[1],
            xs[2] - xs[1],
            xs[3] - xs[1],
            xs[4] - xs[1],
            xs[0] - xs[2],
            xs[1] - xs[2],
            xs[3] - xs[2],
            xs[4] - xs[2],
            xs[0] - xs[3],
            xs[1] - xs[3],
            xs[2] - xs[3],
            xs[0] - xs[4],
            xs[1] - xs[4],
            xs[2] - xs[4],
        ])
        np.testing.assert_almost_equal(
            fem_data.calculate_edge_differences(
                xs, mode='elemental', include_self_loop=False),
            expected_differences)

    def test_aggregate_elemental_edges_to_vertices(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/graph_tet1',
            read_npy=False, read_mesh_only=True, save=False)
        xs = fem_data.convert_nodal2elemental('NODE', calc_average=True)
        expected_differences = np.array([
            np.sum([
                xs[1] - xs[0],
                xs[2] - xs[0],
                xs[3] - xs[0],
                xs[4] - xs[0],
            ], axis=0),
            np.sum([
                xs[0] - xs[1],
                xs[2] - xs[1],
                xs[3] - xs[1],
                xs[4] - xs[1],
            ], axis=0),
            np.sum([
                xs[0] - xs[2],
                xs[1] - xs[2],
                xs[3] - xs[2],
                xs[4] - xs[2],
            ], axis=0),
            np.sum([
                xs[0] - xs[3],
                xs[1] - xs[3],
                xs[2] - xs[3],
            ], axis=0),
            np.sum([
                xs[0] - xs[4],
                xs[1] - xs[4],
                xs[2] - xs[4],
            ], axis=0)
        ])
        np.testing.assert_almost_equal(
            fem_data.aggregate_edges_to_vertices(
                fem_data.calculate_edge_differences(
                    xs, mode='elemental', include_self_loop=False),
                mode='elemental', include_self_loop=False),
            expected_differences)

    def test_calculate_gradient_adjacency_matrix_with_moment_matrix(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_orthogonal_double',
            read_npy=False, save=False)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True)

        x_grad = np.concatenate(
            [g.dot(fem_data.nodes.data[:, [0]]) for g in grads], axis=-1)
        desired = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
        ])
        np.testing.assert_almost_equal(x_grad, desired)

        z_grad = np.concatenate(
            [g.dot(fem_data.nodes.data[:, [2]]) for g in grads], axis=-1)
        desired = np.array([
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
        ])
        np.testing.assert_almost_equal(z_grad, desired)

    def test_calculate_gradient_adjacency_matrix_with_moment_matrix_cube(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/tet2_cube',
            read_npy=False, save=False)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True)

        filter_ = fem_data.filter_first_order_nodes()
        n = np.sum(filter_)
        x_grad = np.stack(
            [g.dot(fem_data.nodes.data[filter_, [0]]) for g in grads], axis=-1)
        desired_x_grad = np.stack([
            np.ones(n), np.zeros(n), np.zeros(n)], axis=-1)
        np.testing.assert_almost_equal(x_grad, desired_x_grad)

        z_grad = np.stack(
            [g.dot(fem_data.nodes.data[filter_, [2]]) for g in grads], axis=-1)
        desired_z_grad = np.stack([
            np.zeros(n), np.zeros(n), np.ones(n)], axis=-1)
        np.testing.assert_almost_equal(z_grad, desired_z_grad)
        # fd = fem_data.to_first_order()
        # fd.nodal_data.update_data(fd.nodes.ids, {'z_grad': z_grad})
        # fd.write('ucd', 'w_moment.inp')

    def test_calculate_gradient_adjacency_matrix_neumann(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/tet2_cube', read_npy=False, save=False)
        grads_wo_neumann \
            = fem_data.calculate_spatial_gradient_adjacency_matrices(
                mode='nodal', n_hop=1, moment_matrix=True, normals=None)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True, normals=True)

        inversed_moment_tensors = fem_data.nodal_data.get_attribute_data(
            'inversed_moment_tensors')
        normals = fem_data.nodal_data.get_attribute_data(
            'filtered_surface_normals')

        filter_ = fem_data.filter_first_order_nodes()
        n = np.sum(filter_)
        x = fem_data.nodes.data[filter_]

        phi = - np.sin(x[:, 0] * 10. + 2 * x[:, 1] * 10.) / 10. + x[:, 2]
        desired_phi_grad = np.stack([
            - np.cos(x[:, 0] * 10. + 2 * x[:, 1] * 10.),
            - np.cos(x[:, 0] * 10. + 2 * x[:, 1] * 10.) * 2,
            np.ones(n)], axis=-1)
        neumann_phi = np.einsum('ij,ij->i', normals, desired_phi_grad)
        neumann_normas = np.einsum('ij,i->ij', normals, neumann_phi)

        phi_grad_wo_neumann = np.stack(
            [g.dot(phi) for g in grads_wo_neumann], axis=-1)
        phi_grad = np.stack(
            [g.dot(phi) for g in grads], axis=-1) + np.einsum(
                'ijk,ik->ij', inversed_moment_tensors, neumann_normas)
        np.testing.assert_almost_equal(
            phi_grad, desired_phi_grad, decimal=1)

        error_phi_grad = phi_grad - desired_phi_grad
        error_phi_grad_wo_neumann = phi_grad_wo_neumann - desired_phi_grad
        error_norm = np.mean(
            np.linalg.norm(error_phi_grad, axis=1))
        error_norm_wo_phi = np.mean(
            np.linalg.norm(error_phi_grad_wo_neumann, axis=1))
        self.assertLess(error_norm, error_norm_wo_phi)

        fd = fem_data.to_first_order()
        fd.nodal_data.update_data(
            fd.nodes.ids, {
                'phi_grad': phi_grad,
                'phi_grad_wo_neumann': phi_grad_wo_neumann, 'phi': phi,
                'desired_phi_grad': desired_phi_grad,
                'error_phi_grad': error_phi_grad,
                'error_phi_grad_wo_neumann': error_phi_grad_wo_neumann,
            })
        fd.write(
            'ucd', 'tests/data/ucd/write_neumann/neumann.inp', overwrite=True)

    def test_calculate_gradient_adjacency_matrix_neumann_normal_weight(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/tet2_cube', read_npy=False, save=False)
        grads_wo_neumann \
            = fem_data.calculate_spatial_gradient_adjacency_matrices(
                mode='nodal', n_hop=1, moment_matrix=True, normals=None)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True, normals=True,
            normal_weight_factor=1., consider_volume=False)

        inversed_moment_tensors = fem_data.nodal_data.get_attribute_data(
            'inversed_moment_tensors')
        normals = fem_data.nodal_data.get_attribute_data(
            'filtered_surface_normals')
        weighted_normals = fem_data.nodal_data.get_attribute_data(
            'weighted_surface_normals')

        filter_ = fem_data.filter_first_order_nodes()
        n = np.sum(filter_)
        x = fem_data.nodes.data[filter_]

        phi = - np.sin(x[:, 0] * 10. + 2 * x[:, 1] * 10.) / 10. + x[:, 2]
        desired_phi_grad = np.stack([
            - np.cos(x[:, 0] * 10. + 2 * x[:, 1] * 10.),
            - np.cos(x[:, 0] * 10. + 2 * x[:, 1] * 10.) * 2,
            np.ones(n)], axis=-1)
        neumann_phi = np.einsum('ij,ij->i', normals, desired_phi_grad)
        neumann_normas = np.einsum('ij,i->ij', weighted_normals, neumann_phi)

        phi_grad_wo_neumann = np.stack(
            [g.dot(phi) for g in grads_wo_neumann], axis=-1)
        phi_grad = np.stack(
            [g.dot(phi) for g in grads], axis=-1) + np.einsum(
                'ijk,ik->ij', inversed_moment_tensors, neumann_normas)
        np.testing.assert_almost_equal(
            phi_grad, desired_phi_grad, decimal=1)

        error_phi_grad = phi_grad - desired_phi_grad
        error_phi_grad_wo_neumann = phi_grad_wo_neumann - desired_phi_grad
        error_norm = np.mean(
            np.linalg.norm(error_phi_grad, axis=1))
        error_norm_wo_phi = np.mean(
            np.linalg.norm(error_phi_grad_wo_neumann, axis=1))
        self.assertLess(error_norm, error_norm_wo_phi)

        fd = fem_data.to_first_order()
        fd.nodal_data.update_data(
            fd.nodes.ids, {
                'phi_grad': phi_grad,
                'phi_grad_wo_neumann': phi_grad_wo_neumann, 'phi': phi,
                'weighted_surface_normals': normals,
                'desired_phi_grad': desired_phi_grad,
                'error_phi_grad': error_phi_grad,
                'error_phi_grad_wo_neumann': error_phi_grad_wo_neumann,
            })
        fd.write(
            'ucd',
            'tests/data/ucd/write_neumann_factor/neumann.inp', overwrite=True)

    def test_calculate_gradient_adjacency_matrix_normals(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_plane', read_npy=False, save=False)
        normals = np.array([
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
        ])
        with self.assertRaises(np.linalg.LinAlgError):
            fem_data.calculate_spatial_gradient_adjacency_matrices(
                mode='elemental', n_hop=1, moment_matrix=True, normals=None)
        grads_w_normals \
            = fem_data.calculate_spatial_gradient_adjacency_matrices(
                mode='elemental', n_hop=1, moment_matrix=True, normals=normals)

        pos = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 1., 0.],
        ])

        np.testing.assert_almost_equal(
            grads_w_normals[0].toarray(),
            np.array([
                [-0.75,  0.75, -0.25,  0.25],
                [-0.75,  0.75, -0.25,  0.25],
                [-0.25,  0.25, -0.75,  0.75],
                [-0.25,  0.25, -0.75,  0.75]]))
        np.testing.assert_almost_equal(
            grads_w_normals[1].toarray(),
            np.array([
                [-0.75, -0.25,  0.75,  0.25],
                [-0.25, -0.75,  0.25,  0.75],
                [-0.75, -0.25,  0.75,  0.25],
                [-0.25, -0.75,  0.25,  0.75]]))
        np.testing.assert_almost_equal(
            grads_w_normals[2].toarray(),
            np.array([
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]))

        x_grad = np.stack(
            [g.dot(pos[:, 0]) for g in grads_w_normals], axis=-1)
        y_grad = np.stack(
            [g.dot(pos[:, 1]) for g in grads_w_normals], axis=-1)
        np.testing.assert_almost_equal(x_grad, np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
        ]))
        np.testing.assert_almost_equal(y_grad, np.array([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
        ]))

    def test_calculate_gradient_adjacency_matrix_with_moment_matrix_hex(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/cube',
            read_npy=False, save=False)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True)

        filter_ = fem_data.filter_first_order_nodes()
        n = np.sum(filter_)
        x_grad = np.stack(
            [g.dot(fem_data.nodes.data[filter_, [0]]) for g in grads],
            axis=-1)
        desired_x_grad = np.stack([
            np.ones(n), np.zeros(n), np.zeros(n)], axis=-1)
        np.testing.assert_almost_equal(x_grad, desired_x_grad)

        z_grad = np.stack(
            [g.dot(fem_data.nodes.data[filter_, [2]]) for g in grads], axis=-1)
        desired_z_grad = np.stack([
            np.zeros(n), np.zeros(n), np.ones(n)], axis=-1)
        np.testing.assert_almost_equal(z_grad, desired_z_grad)

    def test_calculate_frame_tensor_adjs_node_rank1_tet_orthogonal(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_orthogonal',
            read_npy=False, save=False)

        frame_tensor_adjs = fem_data.calculate_frame_tensor_adjs(
            mode='nodal', n_hop=1, tensor_rank=1)
        frame_tensor_array = np.transpose(
            np.stack([
                [s.toarray() for s in ss] for ss in frame_tensor_adjs]),
            (-2, -1, 0, 1))
        n = frame_tensor_array.shape[0]
        ids = np.stack([np.eye(3) for _ in range(n)])
        np.testing.assert_almost_equal(np.sum(frame_tensor_array, axis=1), ids)
        vectors = np.random.rand(n, 3)
        np.testing.assert_almost_equal(
            np.einsum('ijkl,il->ik', frame_tensor_array, vectors), vectors)

    def test_calculate_frame_tensor_adjs_node_rank1_tet2_cube(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3_distorted',
            read_npy=False, save=False)
        frame_tensor_adjs = fem_data.calculate_frame_tensor_adjs(
            mode='nodal', n_hop=2, tensor_rank=1)
        frame_tensor_array = np.transpose(
            np.stack([
                [s.toarray() for s in ss] for ss in frame_tensor_adjs]),
            (-2, -1, 0, 1))
        n = frame_tensor_array.shape[0]
        ids = np.stack([np.eye(3) for _ in range(n)])
        np.testing.assert_almost_equal(np.sum(frame_tensor_array, axis=1), ids)
        vectors = np.random.rand(n, 3)
        np.testing.assert_almost_equal(
            np.einsum('ijkl,il->ik', frame_tensor_array, vectors), vectors)

    def test_calculate_frame_tensor_adjs_node_rank2_tet(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3_distorted',
            read_npy=False, save=False)

        frame_tensor_adjs = fem_data.calculate_frame_tensor_adjs(
            mode='nodal', n_hop=1, tensor_rank=1)
        frame_tensor_array = np.transpose(
            np.stack([
                [s.toarray() for s in ss] for ss in frame_tensor_adjs]),
            (-2, -1, 0, 1))
        n = frame_tensor_array.shape[0]
        tensors = np.random.rand(n, 3, 3)
        np.testing.assert_almost_equal(
            np.einsum(
                'ijst,ikuv,itv->isu',
                frame_tensor_array, frame_tensor_array, tensors), tensors)

    def test_calculate_inner_product_adj(self):
        fem_data = FEMData()
        rank2_1 = np.random.rand(10, 3, 3)
        rank2_2 = np.random.rand(10, 10, 3, 3)
        rank2_2_csr = [
            [sp.csr_matrix(rank2_2[:, :, i, j]) for j in range(3)]
            for i in range(3)]
        inner_prod = fem_data._calculate_inner_product_adj(
            rank2_1, rank2_2_csr)
        np.testing.assert_almost_equal(
            inner_prod.toarray(), np.einsum('ikl,ijkl->ij', rank2_1, rank2_2))

    def test_inverse_tensors_rank4(self):
        fem_data = FEMData()
        rank4 = np.random.rand(10, 3, 3, 3, 3)
        inverted_rank4 = fem_data._inverse_tensors(rank4)
        prod = np.einsum('ijklm,ilmno->ijkno', inverted_rank4, rank4)

        def generate_basis(i, j):
            b = np.zeros((10, 3, 3))
            b[:, i, j] = 1.
            return b
        np.testing.assert_almost_equal(prod[:, 0, 0], generate_basis(0, 0))
        np.testing.assert_almost_equal(prod[:, 1, 0], generate_basis(1, 0))
        np.testing.assert_almost_equal(prod[:, 2, 0], generate_basis(2, 0))
        np.testing.assert_almost_equal(prod[:, 0, 1], generate_basis(0, 1))
        np.testing.assert_almost_equal(prod[:, 1, 1], generate_basis(1, 1))
        np.testing.assert_almost_equal(prod[:, 2, 1], generate_basis(2, 1))
        np.testing.assert_almost_equal(prod[:, 0, 2], generate_basis(0, 2))
        np.testing.assert_almost_equal(prod[:, 1, 2], generate_basis(1, 2))
        np.testing.assert_almost_equal(prod[:, 2, 2], generate_basis(2, 2))

    def test_dot_ndarray_sparse_frame_rank_1(self):
        array_for_sparse = np.reshape(np.arange(10 * 10 * 3), (3, 10, 10))
        sparses = [sp.csr_matrix(a) for a in array_for_sparse]
        array = np.reshape(np.arange(10 * 3 * 3), (10, 3, 3))
        fem_data = FEMData()
        actual = np.stack(
            [
                s.toarray() for s
                in fem_data._dot_ndarray_sparse(array, sparses)],
            axis=-1)
        desired = np.einsum('ikl,lij->ijk', array, array_for_sparse)
        np.testing.assert_almost_equal(actual, desired)

    def test_calculate_tensor_power_ndarray_rank2_power3(self):
        fem_data = FEMData()
        rank2 = np.random.rand(10, 3, 3)
        powered = fem_data.calculate_tensor_power(rank2, power=3)
        rank4 = np.einsum('ijk,ilm->ijklm', rank2, rank2)
        rank6 = np.einsum('ijklm,ino->ijklmno', rank4, rank2)
        np.testing.assert_almost_equal(powered, rank6)

    def test_calculate_tensor_power_sparse_rank1_power2(self):
        fem_data = FEMData()
        rank1 = [
            sp.coo_matrix(
                ([1., 0., 0.], ([0, 0, 0], [1, 2, 3])), shape=(4, 4)),
            sp.coo_matrix(
                ([0., 1., 0.], ([0, 0, 0], [1, 2, 3])), shape=(4, 4)),
            sp.coo_matrix(
                ([0., 0., 1.], ([0, 0, 0], [1, 2, 3])), shape=(4, 4)),
        ]
        coo_powered = fem_data.calculate_tensor_power(rank1, power=2)
        prod = np.transpose(np.array([
            [s.toarray() for s in ss] for ss in coo_powered]), (2, 3, 0, 1))

        def generate_basis(i, j):
            b = np.zeros((3, 3))
            b[i, j] = 1.
            return b

        np.testing.assert_almost_equal(prod[0, 0, :, :], np.zeros((3, 3)))
        np.testing.assert_almost_equal(prod[0, 1, :, :], generate_basis(0, 0))
        np.testing.assert_almost_equal(prod[0, 2, :, :], generate_basis(1, 1))
        np.testing.assert_almost_equal(prod[0, 3, :, :], generate_basis(2, 2))

    def test_calculate_tensor_power_sparse_rank2_power2(self):
        fem_data = FEMData()
        rank2 = np.random.rand(10, 10, 3, 3)
        rank4 = np.einsum('ijkl,ijmn->ijklmn', rank2, rank2)

        rank2_coo = [
            [sp.coo_matrix(rank2[:, :, i, j]) for j in range(3)]
            for i in range(3)]
        coo_powered = fem_data.calculate_tensor_power(rank2_coo, power=2)
        coo_powered_to_numpy = np.transpose(np.stack([
            [[[s.toarray() for s in ss] for ss in sss] for sss in ssss]
            for ssss in coo_powered]), (-2, -1, 0, 1, 2, 3))
        np.testing.assert_almost_equal(coo_powered_to_numpy, rank4)

        rank2_csr = [
            [sp.csr_matrix(rank2[:, :, i, j]) for j in range(3)]
            for i in range(3)]
        csr_powered = fem_data.calculate_tensor_power(rank2_csr, power=2)
        csr_powered_to_numpy = np.transpose(np.stack([
            [[[s.toarray() for s in ss] for ss in sss] for sss in ssss]
            for ssss in csr_powered]), (-2, -1, 0, 1, 2, 3))
        np.testing.assert_almost_equal(csr_powered_to_numpy, rank4)

    def test_calculate_tensor_power_sparse_rank1_power4(self):
        fem_data = FEMData()
        rank1 = np.random.rand(10, 10, 3)
        rank2 = np.einsum('ijk,ijl->ijkl', rank1, rank1)
        rank4 = np.einsum('ijkl,ijmn->ijklmn', rank2, rank2)

        rank1_coo = [sp.coo_matrix(rank1[:, :, i]) for i in range(3)]
        coo_powered = fem_data.calculate_tensor_power(rank1_coo, power=4)
        coo_powered_to_numpy = np.transpose(np.stack([
            [[[s.toarray() for s in s_] for s_ in s__] for s__ in s___]
            for s___ in coo_powered]), (4, 5, 0, 1, 2, 3))
        np.testing.assert_almost_equal(coo_powered_to_numpy, rank4)

        rank1_csr = [sp.csr_matrix(rank1[:, :, i]) for i in range(3)]
        csr_powered = fem_data.calculate_tensor_power(rank1_csr, power=4)
        csr_powered_to_numpy = np.transpose(np.stack([
            [[[s.toarray() for s in s_] for s_ in s__] for s__ in s___]
            for s___ in csr_powered]), (4, 5, 0, 1, 2, 3))
        np.testing.assert_almost_equal(csr_powered_to_numpy, rank4)

    def test_calculate_gradient_adjacency_matrix_with_moment_matrix_mix(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/mix_hex_hexprism',
            read_npy=False, save=False)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True)

        filter_ = fem_data.filter_first_order_nodes()
        n = np.sum(filter_)
        x_grad = np.stack(
            [g.dot(fem_data.nodes.data[filter_, [0]]) for g in grads], axis=-1)
        desired_x_grad = np.stack([
            np.ones(n), np.zeros(n), np.zeros(n)], axis=-1)
        np.testing.assert_almost_equal(x_grad, desired_x_grad)

        z_grad = np.stack(
            [g.dot(fem_data.nodes.data[filter_, [2]]) for g in grads], axis=-1)
        desired_z_grad = np.stack([
            np.zeros(n), np.zeros(n), np.ones(n)], axis=-1)
        np.testing.assert_almost_equal(z_grad, desired_z_grad)
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {'z_grad': z_grad})
        fem_data.nodal_data.pop('inversed_moment_tensors')
        fem_data.write(
            'vtk', 'tests/data/vtk/write_w_moment_hex_prism/mesh.vtk',
            overwrite=True)

    def test_calculate_gradient_adjacency_matrix_neumann_mix(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/mix_hex_hexprism',
            read_npy=False, save=False)
        grads_wo_neumann \
            = fem_data.calculate_spatial_gradient_adjacency_matrices(
                mode='nodal', n_hop=1, moment_matrix=True, normals=None)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True, normals=True)

        inversed_moment_tensors = fem_data.nodal_data.get_attribute_data(
            'inversed_moment_tensors')
        normals = fem_data.nodal_data.get_attribute_data(
            'filtered_surface_normals')

        filter_ = fem_data.filter_first_order_nodes()
        n = np.sum(filter_)
        x = fem_data.nodes.data[filter_]

        scale = 1.
        phi = - np.sin(x[:, 0] / scale + 2 * x[:, 1] / scale) * scale + x[:, 2]
        desired_phi_grad = np.stack([
            - np.cos(x[:, 0] / scale + 2 * x[:, 1] / scale),
            - np.cos(x[:, 0] / scale + 2 * x[:, 1] / scale) * 2,
            np.ones(n)], axis=-1)
        neumann_phi = np.einsum('ij,ij->i', normals, desired_phi_grad)
        neumann_normas = np.einsum('ij,i->ij', normals, neumann_phi)

        phi_grad_wo_neumann = np.stack(
            [g.dot(phi) for g in grads_wo_neumann], axis=-1)
        phi_grad = np.stack(
            [g.dot(phi) for g in grads], axis=-1) + np.einsum(
                'ijk,ik->ij', inversed_moment_tensors, neumann_normas)
        np.testing.assert_almost_equal(
            phi_grad, desired_phi_grad, decimal=0)

        error_phi_grad = phi_grad - desired_phi_grad
        error_phi_grad_wo_neumann = phi_grad_wo_neumann - desired_phi_grad
        error_norm = np.mean(
            np.linalg.norm(error_phi_grad, axis=1))
        error_norm_wo_phi = np.mean(
            np.linalg.norm(error_phi_grad_wo_neumann, axis=1))
        self.assertLess(error_norm, error_norm_wo_phi)

        fem_data.nodal_data.pop('inversed_moment_tensors')
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {
                'phi_grad': phi_grad,
                'phi_grad_wo_neumann': phi_grad_wo_neumann, 'phi': phi,
                'desired_phi_grad': desired_phi_grad,
                'error_phi_grad': error_phi_grad,
                'error_phi_grad_wo_neumann': error_phi_grad_wo_neumann,
            })
        fem_data.write(
            'vtk', 'tests/data/vtk/write_w_moment_hex_prism_neumann/mesh.vtk',
            overwrite=True)

    def test_calculate_spatial_gradient_incidence_matrix_w_moment(self):
        fem_data = brick_generator.generate_brick('hex', 10, 5, 3)
        g_inc, int_inc = fem_data.calculate_spatial_gradient_incidence_matrix(
            mode='nodal', moment_matrix=True, consider_volume=False)
        g_adj = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True, consider_volume=False)
        inversed_moment_tensors = fem_data.nodal_data.get_attribute_data(
            'inversed_moment_tensors')

        for i in range(3):
            np.testing.assert_almost_equal(
                np.einsum(
                    'il,ijl->ij', inversed_moment_tensors[:, i, :],
                    np.stack(
                        [int_inc.dot(g_inc[k]).toarray() for k in range(3)],
                        axis=-1)),
                g_adj[i].toarray())

        filter_ = fem_data.filter_first_order_nodes()

        inc_x_grad = np.einsum(
            'ijk,ik->ij', inversed_moment_tensors, np.stack([
                int_inc.dot(g.dot(fem_data.nodes.data[filter_, [0]]))
                for g in g_inc], axis=-1))
        desired_x_grad = np.stack([
            g.dot(fem_data.nodes.data[filter_, [0]])
            for g in g_adj], axis=-1)
        np.testing.assert_almost_equal(inc_x_grad, desired_x_grad)

        inc_z_grad = np.einsum(
            'ijk,ik->ij', inversed_moment_tensors, np.stack([
                int_inc.dot(g.dot(fem_data.nodes.data[filter_, [2]]))
                for g in g_inc], axis=-1))
        desired_z_grad = np.stack([
            g.dot(fem_data.nodes.data[filter_, [2]])
            for g in g_adj], axis=-1)
        np.testing.assert_almost_equal(inc_z_grad, desired_z_grad)

        r = np.sin(fem_data.nodes.data[filter_, [1]])
        inc_r_grad = np.einsum(
            'ijk,ik->ij', inversed_moment_tensors,
            np.stack([int_inc.dot(g.dot(r)) for g in g_inc], axis=-1))
        desired_r_grad = np.stack([g.dot(r) for g in g_adj], axis=-1)
        np.testing.assert_almost_equal(inc_r_grad, desired_r_grad)

    def test_calculate_spatial_gradient_incidence_matrix_wo_moment(self):
        fem_data = brick_generator.generate_brick('hex', 10, 5, 3)
        g_inc, int_inc = fem_data.calculate_spatial_gradient_incidence_matrix(
            mode='nodal', moment_matrix=False, consider_volume=False)
        g_adj = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=False, consider_volume=False)

        for i in range(3):
            np.testing.assert_almost_equal(
                int_inc.dot(g_inc[i]).toarray(), g_adj[i].toarray())

        filter_ = fem_data.filter_first_order_nodes()

        inc_x_grad = np.stack([
            int_inc.dot(g.dot(fem_data.nodes.data[filter_, [0]]))
            for g in g_inc], axis=-1)
        desired_x_grad = np.stack([
            g.dot(fem_data.nodes.data[filter_, [0]])
            for g in g_adj], axis=-1)
        np.testing.assert_almost_equal(inc_x_grad, desired_x_grad)

        inc_z_grad = np.stack([
            int_inc.dot(g.dot(fem_data.nodes.data[filter_, [2]]))
            for g in g_inc], axis=-1)
        desired_z_grad = np.stack([
            g.dot(fem_data.nodes.data[filter_, [2]])
            for g in g_adj], axis=-1)
        np.testing.assert_almost_equal(inc_z_grad, desired_z_grad)

        r = np.sin(fem_data.nodes.data[filter_, [1]])
        inc_r_grad = np.stack([int_inc.dot(g.dot(r)) for g in g_inc], axis=-1)
        desired_r_grad = np.stack([g.dot(r) for g in g_adj], axis=-1)
        np.testing.assert_almost_equal(inc_r_grad, desired_r_grad)

    def test_spatial_gradient_incidence_neumann_mix(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/mix_hex_hexprism',
            read_npy=False, save=False)

        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True, normals=True)
        inversed_moment_tensors = fem_data.nodal_data.get_attribute_data(
            'inversed_moment_tensors')
        normals = fem_data.nodal_data.get_attribute_data(
            'weighted_surface_normals')

        filter_ = fem_data.filter_first_order_nodes()
        n = np.sum(filter_)
        x = fem_data.nodes.data[filter_]

        scale = 1.
        phi = - np.sin(x[:, 0] / scale + 2 * x[:, 1] / scale) * scale + x[:, 2]
        desired_phi_grad = np.stack([
            - np.cos(x[:, 0] / scale + 2 * x[:, 1] / scale),
            - np.cos(x[:, 0] / scale + 2 * x[:, 1] / scale) * 2,
            np.ones(n)], axis=-1)
        neumann_phi = np.einsum('ij,ij->i', normals, desired_phi_grad)
        neumann_normas = np.einsum('ij,i->ij', normals, neumann_phi)

        isoam_phi_grad = np.stack(
            [g.dot(phi) for g in grads], axis=-1) + np.einsum(
                'ijk,ik->ij', inversed_moment_tensors, neumann_normas)

        fem_data.nodal_data.pop('inversed_moment_tensors')
        fem_data.nodal_data.pop('weighted_surface_normals')

        sgms, eim = fem_data.calculate_spatial_gradient_incidence_matrix(
            mode='nodal', n_hop=1, moment_matrix=True, normals=True)
        inversed_moment_tensors = fem_data.nodal_data.get_attribute_data(
            'inversed_moment_tensors')
        normals = fem_data.nodal_data.get_attribute_data(
            'weighted_surface_normals')

        isoim_phi_grad = np.einsum(
            'ijk,ik->ij', inversed_moment_tensors, np.stack(
                [eim.dot(sgm.dot(phi)) for sgm in sgms], axis=-1)
        ) + np.einsum(
            'ijk,ik->ij', inversed_moment_tensors, neumann_normas)

        np.testing.assert_almost_equal(
            isoim_phi_grad, isoam_phi_grad, decimal=0)

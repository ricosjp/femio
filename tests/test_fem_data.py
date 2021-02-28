import os
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

# from femio.fem_attribute import FEMAttribute
from femio.fem_data import FEMData
# from femio.fem_elemental_attribute import FEMElementalAttribute


RUN_FISTR = True

FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'


class TestFEMData(unittest.TestCase):

    def test_read_directory(self):
        """Files should be correctly read with read_directory method."""
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False)
        np.testing.assert_almost_equal(
            fem_data.nodal_data['INITIAL_TEMPERATURE'].data,
            np.array([[10.0,
                       20.0,
                       30.0,
                       40.0,
                       50.0,
                       60.0,
                       70.0,
                       80.0,
                       90.0,
                       100.0,
                       110.0,
                       120.0]]).T)
        np.testing.assert_almost_equal(
            fem_data.materials.get_attribute_data('ORIENTATION'),
            np.array([
                [0.024767077205754086, 0.3920046583225272, 0.39343221896213765,
                 -0.6143822108570163, 1.9159004529686463, -1.3560190673207662,
                 0.0, 0.0, 0.0],
                [-0.5507555559853339, 1.0875686817748702, 0.9523091815809357,
                 0.7961043196361649, 1.4996034939516303, 1.5948473551761138,
                 0.0, 0.0, 0.0]
            ])
        )

    def test_read_directory_wo_res(self):
        """Files should be correctly read with read_directory method."""
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_wo_res', read_npy=False)
        np.testing.assert_almost_equal(
            fem_data.nodal_data['INITIAL_TEMPERATURE'].data,
            np.array([[10.0,
                       20.0,
                       30.0,
                       40.0,
                       50.0,
                       60.0,
                       70.0,
                       80.0,
                       90.0,
                       100.0,
                       110.0,
                       120.0]]).T)

    def test_read_directory_empty_directory_error(self):
        with self.assertRaises(ValueError):
            FEMData.read_directory(
                'fistr', 'tests/data/fistr/empty', read_npy=False)

    def test_read_saved_npy(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_id_not_from_1', read_npy=False)
        write_dir_name = Path('tests/data/fistr/write_npy')
        if write_dir_name.exists():
            shutil.rmtree(write_dir_name)
        fem_data.save(write_dir_name)
        npy_fem_data = FEMData.read_npy_directory(write_dir_name)
        npy_fem_data.write(
            'fistr', write_dir_name / 'mesh', overwrite=True)

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1 > /dev/null 2>&1",
                cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data,
                decimal=5)

    def test_to_first_order(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3', read_npy=False, save=False)
        filter_ = fem_data.filter_first_order_nodes()
        first_order_fem_data = fem_data.to_first_order()
        np.testing.assert_almost_equal(
            first_order_fem_data.nodes.data, fem_data.nodes.data[filter_])
        np.testing.assert_array_equal(
            first_order_fem_data.elements.data, fem_data.elements.data[:, :4])
        np.testing.assert_almost_equal(
            first_order_fem_data.nodal_data.get_attribute_data('NodalSTRESS'),
            fem_data.nodal_data.get_attribute_data('NodalSTRESS')[filter_])

    def test_to_surface(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_w_node_inside',
            read_npy=False, save=False)
        surface_fem_data = fem_data.to_surface()
        desired_nodes = np.array([
            [0., 0., 0.],
            [4., 0., 0.],
            [0., 4., 0.],
            [0., 0., 4.],
        ])
        desired_elements = np.array([
            [1, 3, 2],
            [1, 2, 4],
            [1, 4, 3],
            [2, 3, 4],
        ])
        desired_normals = np.array([
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            [1/3**.5, 1/3**.5, 1/3**.5],
        ])
        desired_initial_temperature = np.array([[10., 20., 30., 40.]]).T
        np.testing.assert_almost_equal(
            surface_fem_data.nodes.data, desired_nodes)
        np.testing.assert_array_equal(
            surface_fem_data.elements.data, desired_elements)
        np.testing.assert_almost_equal(
            surface_fem_data.calculate_element_normals(), desired_normals)
        np.testing.assert_almost_equal(
            surface_fem_data.nodal_data.get_attribute_data(
                'INITIAL_TEMPERATURE'), desired_initial_temperature)

    def test_to_surface_mix(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/mix_hex_hexprism',
            read_npy=False, save=False)
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {'INITIAL_TEMPERATURE': np.arange(
                len(fem_data.nodes))[:, None] * 10.})
        surface_fem_data = fem_data.to_surface()
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
            [0.9, 0.1, 0.3],
            [0.0, 0.1, 0.6],
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
        desired_normals = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, -0.7071067811865476, -0.7071067811865476],
            [-0.7071067811865476, -0.7071067811865476, 0.0],
            [0.0, -0.7071067811865475, -0.7071067811865475],
            [0.0, -0.7071067811865476, -0.7071067811865476],
            [0.7071067811865476, -0.7071067811865474, 0.0],
            [-0.7071067811865475, -0.7071067811865475, 0.0],
            [0.7071067811865476, -0.7071067811865475, 0.0],
            [-0.7071067811865476, -0.7071067811865476, 0.0],
            [0.7071067811865476, -0.7071067811865475, 0.0],
            [0.0, -0.7071067811865474, 0.7071067811865476],
            [0.0, -0.7071067811865475, 0.7071067811865476],
            [0.0, -0.7071067811865475, 0.7071067811865476],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])

        np.testing.assert_almost_equal(
            surface_fem_data.nodes.data, desired_nodes)
        self.assertEqual(len(surface_fem_data.elements), 50)
        np.testing.assert_almost_equal(
            surface_fem_data.calculate_element_normals(), desired_normals)
        node_filter = np.ones(len(fem_data.nodes)).astype(bool)
        node_filter[[25, 26, 28, 29, 30, 31, 33, 34]] = False
        np.testing.assert_almost_equal(
            surface_fem_data.nodal_data.get_attribute_data(
                'INITIAL_TEMPERATURE'),
            np.arange(len(fem_data.nodes))[node_filter][:, None] * 10)
        surface_fem_data.write(
            'ucd', 'tests/data/ucd/write_surface_mix_hex_hexprism.inp',
            overwrite=True)

    def test_to_first_order_time_series(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_tet2_3', read_npy=False,
            save=False, time_series=True)
        filter_ = fem_data.filter_first_order_nodes()
        first_order_fem_data = fem_data.to_first_order()
        np.testing.assert_almost_equal(
            first_order_fem_data.nodes.data, fem_data.nodes.data[filter_])
        np.testing.assert_array_equal(
            first_order_fem_data.elements.data, fem_data.elements.data[:, :4])
        np.testing.assert_almost_equal(
            first_order_fem_data.nodal_data.get_attribute_data('TEMPERATURE'),
            fem_data.nodal_data['TEMPERATURE'].loc[filter_].data)

    def test_read_saved_npy_mixed_elements(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid',
            read_npy=False, save=False)
        write_dir_name = 'tests/data/fistr/write_npy_mixture_solid'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.settings = {}
        fem_data.save(write_dir_name)
        npy_fem_data = FEMData.read_npy_directory(write_dir_name)
        npy_fem_data.write(
            'fistr', os.path.join(write_dir_name, 'mesh'), overwrite=True)

        np.testing.assert_array_equal(
            fem_data.elements['hex'].ids,
            npy_fem_data.elements['hex'].ids,
        )
        np.testing.assert_array_equal(
            fem_data.elements['hex'].data,
            npy_fem_data.elements['hex'].data,
        )
        np.testing.assert_array_equal(
            fem_data.elements['tet'].ids,
            npy_fem_data.elements['tet'].ids,
        )
        np.testing.assert_array_equal(
            fem_data.elements['tet'].data,
            npy_fem_data.elements['tet'].data,
        )

        if RUN_FISTR:
            os.system(f"cd {write_dir_name} "
                      + "&& fistr1 > /dev/null 2>&1")
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data,
                decimal=5)

    def test_read_saved_npy_mixture_shell(self):
        fem_data = FEMData.read_directory(
            'obj', 'tests/data/obj/mixture_shell',
            read_npy=False, save=False)
        write_dir_name = 'tests/data/obj/write_npy_mixture_shell'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.save(write_dir_name)
        npy_fem_data = FEMData.read_npy_directory(write_dir_name)
        desired_elements = [
            np.array([1, 2, 3, 4]),
            np.array([2, 3, 5]),
            np.array([3, 6, 7, 4])
        ]
        for e1, e2 in zip(npy_fem_data.elements.data, desired_elements):
            np.testing.assert_array_equal(e1, e2)

    def test_save_constraints(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/cload', read_npy=False,
            save=False)
        write_dir_name = 'tests/data/fistr/cload_save'
        fem_data.save(write_dir_name)
        cload_ids = np.array(
            [1,  4,  7, 10, 13, 16, 19, 22, 25,  1,  4,  7, 10, 13, 16, 19,
             22, 25,  3,  6,  9, 12, 15, 18, 21, 24, 27,  3,  6,  9, 12, 15,
             18, 21, 24, 27])
        np.testing.assert_array_equal(
            np.load(write_dir_name + '/femio_constraints.npz')['cload/ids'],
            cload_ids)

    def test_filter_by_node_ids(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid', read_npy=False)
        node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17])
        cut_fem_data = fem_data.cut_with_node_ids(node_ids)
        desired_node_data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        desired_element_ids = np.array([1, 3, 4])
        desired_element_data = [
            np.array([1, 2, 4, 3, 5, 6, 8, 7]),
            np.array([12, 2, 6, 9]),
            np.array([1, 5, 17, 2, 6, 12]),
        ]
        np.testing.assert_array_equal(
            cut_fem_data.nodes.ids, node_ids)
        np.testing.assert_almost_equal(
            cut_fem_data.nodes.data, desired_node_data)
        np.testing.assert_array_equal(
            cut_fem_data.elements.ids, desired_element_ids)
        for ae, de in zip(cut_fem_data.elements.data, desired_element_data):
            np.testing.assert_array_equal(ae, de)

        desired_displacement = np.array([
            [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
            [-3.5252045e-16, 5.0000000e-01, 5.0000000e-01],
            [-5.0000000e-01, -1.9242906e-14, -5.0000000e-01],
            [-5.0000000e-01, 5.0000000e-01, 4.1809655e-14],
            [-5.0000000e-01, 5.0000000e-01, -1.4303842e-15],
            [-5.0000000e-01, 1.0000000e+00, 5.0000000e-01],
            [-1.0000000e+00, 5.0000000e-01, -5.0000000e-01],
            [-1.0000000e+00, 1.0000000e+00, 4.1069137e-14],
            [-5.0000000e-01, 1.5000000e+00, 1.0000000e+00],
            [5.0000000e-01, 5.0000000e-01, 1.0000000e+00],
            [5.0000000e-01, -6.1058976e-16, 5.0000000e-01],
        ])
        np.testing.assert_almost_equal(
            cut_fem_data.nodal_data.get_attribute_data('DISPLACEMENT'),
            desired_displacement)
        desired_young_modulus = np.array([[10., 30., 40.]]).T
        np.testing.assert_almost_equal(
            cut_fem_data.elemental_data.get_attribute_data('Young_modulus'),
            desired_young_modulus)

    def test_cut_with_element_ids(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid', read_npy=False)
        element_ids = np.array([1, 4, 8])

        cut_fem_data = fem_data.cut_with_element_ids(element_ids)
        desired_node_data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
            [0.0, -1.0, 0.0],
        ])

        desired_node_ids = np.array([
            1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17])
        desired_element_data = [
            np.array([1, 2, 4, 3, 5, 6, 8, 7]),
            np.array([1, 5, 17, 2, 6, 12]),
            np.array([13, 3, 4, 14, 15, 7, 8, 16]),
        ]
        np.testing.assert_array_equal(
            cut_fem_data.nodes.ids, desired_node_ids)
        np.testing.assert_almost_equal(
            cut_fem_data.nodes.data, desired_node_data)
        np.testing.assert_array_equal(
            cut_fem_data.elements.ids, element_ids)
        for ae, de in zip(cut_fem_data.elements.data, desired_element_data):
            np.testing.assert_array_equal(ae, de)

    def test_cut_with_element_type(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid', read_npy=False)

        cut_fem_data = fem_data.cut_with_element_type('hex')
        desired_node_data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
        ])
        desired_node_ids = np.array([
            1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16])
        desired_element_ids = np.array([1, 8])
        desired_element_data = [
            np.array([1, 2, 4, 3, 5, 6, 8, 7]),
            np.array([13, 3, 4, 14, 15, 7, 8, 16]),
        ]
        np.testing.assert_array_equal(
            cut_fem_data.nodes.ids, desired_node_ids)
        np.testing.assert_almost_equal(
            cut_fem_data.nodes.data, desired_node_data)
        np.testing.assert_array_equal(
            cut_fem_data.elements.ids, desired_element_ids)
        for ae, de in zip(cut_fem_data.elements.data, desired_element_data):
            np.testing.assert_array_equal(ae, de)

    def test_remove_useless_nodes(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/useless_nodes_disordered',
            read_npy=False, save=False)
        np.testing.assert_array_equal(
            fem_data.nodes.ids, [2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('t_init'),
            np.array([[2, 3, 4, 5, 6, 7, 8, 9]]).T * 10.)

    def test_read_npy_series_data(self):
        fem_data = FEMData.read_npy_directory('tests/data/npy/temperatures')
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('nodal_t_10'),
            np.array([
                [9.2546418],
                [9.3392005],
                [10.689137],
                [18.477557],
                [15.111514],
                [16.652916]])
        )
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('nodal_t_100'),
            np.array([
                [9.3394346],
                [9.4908689],
                [10.985905],
                [18.361863],
                [15.379292],
                [16.664873]])
        )

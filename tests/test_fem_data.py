import os
from pathlib import Path
import shutil
import subprocess
import unittest
import pytest

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
            [1 / 3**.5, 1 / 3**.5, 1 / 3**.5],
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

    def test_to_surface_wo_polygon_surface(self):
        fem_data = FEMData.read_files(
            'vtu', 'tests/data/vtu/no_polygon_surface/mesh.vtu')
        surface_fem_data = fem_data.to_surface()
        nodes = surface_fem_data.nodes.data

        min_x = np.min(nodes[:, 0])
        max_x = np.max(nodes[:, 0])
        min_y = np.min(nodes[:, 1])
        max_y = np.max(nodes[:, 1])
        min_z = np.min(nodes[:, 2])
        max_z = np.max(nodes[:, 2])

        assert np.all(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        np.abs(nodes[:, 0] - min_x) < 1e-5,
                        np.abs(nodes[:, 0] - max_x) < 1e-5),
                    np.logical_or(
                        np.abs(nodes[:, 1] - min_y) < 1e-5,
                        np.abs(nodes[:, 1] - max_y) < 1e-5)),
                np.logical_or(
                    np.abs(nodes[:, 2] - min_z) < 1e-5,
                    np.abs(nodes[:, 2] - max_z) < 1e-5)),
        )

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
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
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
        cload_ids = np.array([
            1, 4, 7, 10, 13, 16, 19, 22, 25,
            3, 6, 9, 12, 15, 18, 21, 24, 27,
            1, 4, 7, 10, 13, 16, 19, 22, 25,
            3, 6, 9, 12, 15, 18, 21, 24, 27,
        ])
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
            np.array([1, 17, 5, 2, 12, 6]),
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
            np.array([1, 17, 5, 2, 12, 6]),
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

    def test_cut_with_element_ids_polyhedron(self):
        fem_data = FEMData.read_files(
            'polyvtk', 'tests/data/vtu/polyhedron/polyhedron.vtu')
        element_ids = np.array([1, 2])
        cut_fem_data = fem_data.cut_with_element_ids(element_ids)
        desired_node_ids = np.arange(1, 13)
        desired_node_data = fem_data.nodes.data[:12]

        np.testing.assert_array_equal(
            cut_fem_data.nodes.ids, desired_node_ids)
        np.testing.assert_almost_equal(
            cut_fem_data.nodes.data, desired_node_data)
        np.testing.assert_array_equal(
            cut_fem_data.elements.ids, element_ids)

        face_0 = [6, 5, 0, 1, 5, 10, 9, 4, 1, 3, 8, 5, 5, 0, 9,
                  11, 8, 3, 4, 5, 8, 11, 10, 3, 9, 10, 11, 3, 0, 3, 1]
        face_1 = [6, 4, 1, 2, 6, 5, 4, 2, 4, 7, 6, 4, 5, 6,
                  7, 8, 4, 3, 8, 7, 4, 4, 1, 3, 4, 2, 4, 1, 5, 8, 3]
        np.testing.assert_array_equal(
            cut_fem_data.elemental_data['face']['polyhedron'].data[0],
            np.array(face_0)
        )
        np.testing.assert_array_equal(
            cut_fem_data.elemental_data['face']['polyhedron'].data[1],
            np.array(face_1)
        )

        fem_data = FEMData.read_files(
            'polyvtk', 'tests/data/vtu/poly_pyramid/mesh.vtu')
        element_ids = np.array([1, 2])
        cut_fem_data = fem_data.cut_with_element_ids(element_ids)
        desired_node_ids = np.arange(1, 10)
        desired_node_data = fem_data.nodes.data[:9]
        np.testing.assert_array_equal(
            cut_fem_data.nodes.ids, desired_node_ids)
        np.testing.assert_almost_equal(
            cut_fem_data.nodes.data, desired_node_data)
        np.testing.assert_array_equal(
            cut_fem_data.elements.ids, element_ids)
        face = [6, 5, 0, 1, 4, 7, 6, 4, 1, 3, 5, 4, 5, 0, 6,
                8, 5, 3, 4, 4, 5, 8, 7, 3, 6, 7, 8, 3, 0, 3, 1]
        np.testing.assert_array_equal(
            cut_fem_data.elemental_data['face']['polyhedron'].data[0],
            np.array(face)
        )
        fem_data = FEMData.read_files(
            'polyvtk', 'tests/data/vtu/complex/mesh.vtu')
        element_ids = np.array([1, 3])
        cut_fem_data = fem_data.cut_with_element_ids(element_ids)

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

    def test_generate_graph_fem_data(self):
        fem_data = FEMData.read_directory(
            'vtk', 'tests/data/vtk/mix_hex_hexprism',
            read_npy=False, save=False)
        grads = fem_data.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=1, moment_matrix=True)
        graph_fem_data = fem_data.generate_graph_fem_data(grads, mode='nodal')
        graph_fem_data.write(
            'ucd', 'tests/data/ucd/write_graph_fem_data/mesh.inp',
            overwrite=True)
        self.assertEqual(
            len(graph_fem_data.elements), np.max([g.getnnz() for g in grads]))

    def test_create_node_group(self):
        file_name = Path('tests/data/fistr/pyramid/pyramid.msh')
        fem_data = FEMData.read_files('fistr', [file_name])

        selected = fem_data.nodes.data[:, 2] == 0
        fem_data.create_node_group('new_group', selected)
        np.testing.assert_equal(
            fem_data.node_groups['new_group'], np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            fem_data.create_node_group('new_group', selected)

    def test_create_element_group_from_node_group_all(self):
        file_name = Path('tests/data/fistr/pyramid/pyramid.msh')
        fem_data = FEMData.read_files('fistr', [file_name])
        fem_data.create_node_group(
            'node_1234', np.array([True, True, True, True, False]))
        fem_data.create_element_group_from_node_group(
            'element_1234', 'node_1234', 'all')
        np.testing.assert_equal(
            fem_data.element_groups['element_1234'], np.array([1]))
        with pytest.raises(ValueError):
            fem_data.create_element_group_from_node_group(
                'element_1234', 'node_1234', 'all')

    def test_create_element_group_from_node_group_any(self):
        file_name = Path('tests/data/fistr/pyramid/pyramid.msh')
        fem_data = FEMData.read_files('fistr', [file_name])
        fem_data.create_node_group(
            'node_1234', np.array([True, True, True, True, False]))
        fem_data.create_element_group_from_node_group(
            'element_1234', 'node_1234', 'any')
        np.testing.assert_equal(
            fem_data.element_groups['element_1234'], np.array([1, 2]))
        with pytest.raises(ValueError):
            fem_data.create_element_group_from_node_group(
                'element_1234', 'node_1234', 'all')

    def test_resolve_degeneracy_hex2prism(self):
        # prism, hex, hex -> prism, prism, prism
        fem_data = FEMData.read_files(
            'ucd', 'tests/data/ucd/degeneracy_1/mesh.inp')
        new_fem_data = fem_data.resolve_degeneracy()
        volumes = new_fem_data.calculate_element_volumes()

        assert 'hex' in fem_data.elements
        assert 'prism' in fem_data.elements
        assert 'hex' not in new_fem_data.elements
        assert 'prism' in new_fem_data.elements
        np.testing.assert_equal(
            fem_data.elements['prism'].ids, np.array([1]))
        np.testing.assert_equal(
            fem_data.elements['prism'].data, np.array(
                [[2, 4, 5, 7, 9, 10]]))
        np.testing.assert_equal(
            fem_data.elements['hex'].ids, np.array([2, 3]))
        np.testing.assert_equal(
            fem_data.elements['hex'].data, np.array(
                [[1, 1, 4, 3, 6, 6, 9, 8], [1, 2, 2, 4, 6, 7, 7, 9]]))
        np.testing.assert_equal(
            fem_data.elements.ids, np.array([1, 2, 3]))
        np.testing.assert_equal(
            fem_data.elements.types, np.array(['prism', 'hex', 'hex']))

        np.testing.assert_equal(
            new_fem_data.elements['prism'].ids, np.array([1, 2, 3]))
        np.testing.assert_equal(
            new_fem_data.elements['prism'].data, np.array([
                [2, 4, 5, 7, 9, 10],
                [1, 3, 4, 6, 8, 9],
                [1, 4, 2, 6, 9, 7]
            ]))
        np.testing.assert_equal(
            new_fem_data.elements.ids, np.array([1, 2, 3]))
        np.testing.assert_equal(
            new_fem_data.elements.types, np.array(['prism', 'prism', 'prism']))
        np.testing.assert_almost_equal(volumes, .5)

    def test_resolve_degeneracy_no_degeneracy(self):
        # prism, hex -> prism, hex (nodegeneracy)
        fem_data = FEMData.read_files(
            'ucd', 'tests/data/ucd/degeneracy_2/mesh.inp')

        new_fem_data = fem_data.resolve_degeneracy()
        volumes = new_fem_data.calculate_element_volumes()
        assert 'hex' in new_fem_data.elements
        assert 'prism' in new_fem_data.elements
        np.testing.assert_equal(
            new_fem_data.elements['hex'].ids, np.array([2]))
        np.testing.assert_equal(
            new_fem_data.elements['hex'].data, np.array(
                [[1, 2, 4, 3, 6, 7, 9, 8]]))
        np.testing.assert_equal(
            new_fem_data.elements['prism'].ids, np.array([1]))
        np.testing.assert_equal(
            new_fem_data.elements['prism'].data, np.array(
                [[2, 4, 5, 7, 9, 10]]))
        np.testing.assert_equal(
            new_fem_data.elements.ids, np.array([1, 2]))
        np.testing.assert_equal(
            new_fem_data.elements.types, np.array(['prism', 'hex']))
        np.testing.assert_almost_equal(volumes[:, 0], [.5, 1.])

    def test_resolve_degeneracy_hex2prism_2(self):
        # hex, hex, hex -> hex, prism, prism
        fem_data = FEMData.read_files(
            'ucd', 'tests/data/ucd/degeneracy_3/mesh.inp')

        new_fem_data = fem_data.resolve_degeneracy()
        volumes = new_fem_data.calculate_element_volumes()
        assert 'hex' in fem_data.elements
        assert 'prism' not in fem_data.elements
        assert 'hex' in new_fem_data.elements
        assert 'prism' in new_fem_data.elements
        np.testing.assert_equal(
            fem_data.elements['hex'].ids, np.array([1, 2, 3]))
        np.testing.assert_equal(
            fem_data.elements['hex'].data, np.array([
                [1, 2, 5, 4, 7, 8, 11, 10],
                [2, 3, 5, 5, 8, 9, 11, 11],
                [3, 6, 5, 3, 9, 12, 11, 9]]))
        np.testing.assert_equal(
            fem_data.elements.ids, np.array([1, 2, 3]))
        np.testing.assert_equal(
            fem_data.elements.types, np.array(['hex', 'hex', 'hex']))

        np.testing.assert_equal(
            new_fem_data.elements['prism'].ids, np.array([2, 3]))
        np.testing.assert_equal(
            new_fem_data.elements['prism'].data, np.array([
                [2, 5, 3, 8, 11, 9], [3, 5, 6, 9, 11, 12]]))
        np.testing.assert_equal(
            new_fem_data.elements['hex'].ids, np.array([1]))
        np.testing.assert_equal(
            new_fem_data.elements['hex'].data, np.array([
                [1, 2, 5, 4, 7, 8, 11, 10]]))
        np.testing.assert_equal(
            new_fem_data.elements.ids, np.array([1, 2, 3]))
        np.testing.assert_equal(
            new_fem_data.elements.types, np.array(['hex', 'prism', 'prism']))
        np.testing.assert_almost_equal(volumes[:, 0], [1., .5, .5])

    def test_to_polyhedron(self):
        def check_face_format(face):
            directed_edges = []
            n = face[0]
            ptr = 1
            for _ in range(n):
                k = face[ptr]
                ptr += 1
                F = face[ptr:ptr + k]
                ptr += k
                for i in range(len(F)):
                    a = F[i - 1]
                    b = F[i]
                    directed_edges.append((a, b))
            if ptr != len(face):
                return False
            se = set(directed_edges)
            if len(directed_edges) != len(se):
                return False
            for a, b in directed_edges:
                if (b, a) not in se:
                    return False
            return True

        fem_data = FEMData.read_files(
            'vtk', 'tests/data/vtk/tet_3/mesh.vtk')
        new_fem_data = fem_data.to_polyhedron()
        np.testing.assert_equal(new_fem_data.elements.data, np.array(
            [5, 6, 7, 4, 1, 2, 3, 7, 1, 6, 7, 5, 1, 2, 7, 6]).reshape(4, 4))
        face_data = new_fem_data.elemental_data['face']['polyhedron'].data
        assert isinstance(face_data[0], list)
        np.testing.assert_equal(
            face_data[0], [
                4, 3, 4, 6, 5, 3, 3, 4, 5, 3, 3, 6, 4, 3, 3, 5, 6])
        for face in face_data:
            assert check_face_format(face)

        fem_data = FEMData.read_files(
            'vtk', 'tests/data/vtk/hex/mesh.vtk')
        new_fem_data = fem_data.to_polyhedron()
        np.testing.assert_equal(new_fem_data.elements.data, np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(2, 8))
        face_data = new_fem_data.elemental_data['face']['polyhedron'].data
        assert isinstance(face_data[0], list)
        desired = [6, 4, 4, 5, 6, 7, 4, 5, 4, 0, 1, 4, 6, 5,
                   1, 2, 4, 7, 6, 2, 3, 4, 4, 7, 3, 0, 4, 3, 2, 1, 0]
        np.testing.assert_equal(
            face_data[0], desired)
        for face in face_data:
            assert check_face_format(face)

        fem_data = FEMData.read_files(
            'ucd', 'tests/data/ucd/prism/mesh.inp')
        new_fem_data = fem_data.to_polyhedron()
        np.testing.assert_equal(new_fem_data.elements.data, np.array(
            [204, 2042, 2043, 136, 1973, 1974]).reshape(1, 6))
        face_data = new_fem_data.elemental_data['face']['polyhedron'].data
        assert isinstance(face_data[0], list)
        desired = [5, 3, 1, 4, 5, 3, 3, 2, 0, 4, 4,
                   1, 0, 2, 4, 5, 4, 2, 3, 4, 1, 5, 3, 0]
        np.testing.assert_equal(
            face_data[0], desired)
        for face in face_data:
            assert check_face_format(face)

        fem_data = FEMData.read_files(
            'polyvtk', 'tests/data/vtu/pyramid/pyramid.vtu')
        new_fem_data = fem_data.to_polyhedron()
        face_data = new_fem_data.elemental_data['face']['polyhedron'].data
        assert isinstance(face_data[1], list)
        desired = [5, 4, 7, 6, 5, 4, 3, 4, 5, 8,
                   3, 5, 6, 8, 3, 6, 7, 8, 3, 7, 4, 8]
        np.testing.assert_equal(
            face_data[1], desired)
        for face in face_data:
            assert check_face_format(face)

    def test_to_csr(self):
        fem_data = FEMData.read_files(
            'polyvtk', 'tests/data/vtu/polyhedron/polyhedron.vtu')
        faces = fem_data.elemental_data['face']['polyhedron'].data
        csr = fem_data.face_data_csr()
        assert len(csr) == 2
        indptr, dat = csr
        assert len(indptr) == len(faces) + 1
        assert indptr[-1] == len(dat)
        for n in range(len(faces)):
            desired = np.array(faces[n])
            actual = dat[indptr[n]: indptr[n + 1]]
            np.testing.assert_array_equal(desired, actual)

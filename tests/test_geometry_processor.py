import pathlib
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

    def test_calculate_volume_tet(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume', read_npy=False,
            save=False)
        actual_volumes = fem_data.calculate_element_volumes()
        desired_volumes = np.array([[1 / 6], [1 / 2], [1 / 3], [1.]])
        np.testing.assert_almost_equal(actual_volumes, desired_volumes)

    def test_calculate_volume_hex(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_cross', read_npy=False,
            save=False)
        actual_volumes = fem_data.calculate_element_volumes()
        desired_volumes = np.ones((7, 1)) * 8.
        np.testing.assert_almost_equal(actual_volumes, desired_volumes)

    def test_calculate_volume_tet_negative_raise_valueerror(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False)
        with self.assertRaises(ValueError):
            fem_data.calculate_element_volumes(
                raise_negative_volume=True)

    def test_calculate_volume_tet_negative_absolute(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_volume_negative', read_npy=False,
            save=False)
        actual_volumes = fem_data.calculate_element_volumes(
            raise_negative_volume=False, return_abs_volume=True)
        desired_volumes = np.array([[1 / 6], [1 / 2], [1 / 3], [1.]])
        np.testing.assert_almost_equal(actual_volumes, desired_volumes)

    def test_calculate_area_tri(self):
        data_directory = 'tests/data/stl/area'
        fem_data = FEMData.read_directory(
            'stl', data_directory, read_npy=False, save=False)

        desired_areas = np.array([
            [3.],
            [6.],
        ])
        fem_data.calculate_element_areas()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('area'), desired_areas)

    def test_calculate_area_quad(self):
        data_directory = 'tests/data/obj/quad'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)
        desired_areas = np.array([
            [1.],
            [1.5],
            [np.sqrt(2)],
        ])
        fem_data.calculate_element_areas()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('area'), desired_areas)

    def test_calculate_normal_tri(self):
        data_directory = 'tests/data/stl/multiple'
        fem_data = FEMData.read_directory(
            'stl', data_directory, read_npy=False, save=False)

        desired_normals = np.array([
            [0., 0., 1.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        fem_data.calculate_element_normals()
        np.testing.assert_almost_equal(
            fem_data.extract_direction_feature(
                fem_data.elemental_data.get_attribute_data('normal')),
            fem_data.extract_direction_feature(desired_normals), decimal=5)

    def test_calculate_normal_quad(self):
        data_directory = 'tests/data/obj/quad'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_normals = np.array([
            [0., 0., 1.],
            [0., 0., 1.],
            [-np.sqrt(.5), -np.sqrt(.5), 0.],
        ])
        fem_data.calculate_element_normals()
        np.testing.assert_almost_equal(
            fem_data.extract_direction_feature(
                fem_data.elemental_data.get_attribute_data('normal')),
            fem_data.extract_direction_feature(desired_normals), decimal=5)

    def test_calculate_surface_normals(self):
        data_directory = 'tests/data/vtk/tet2_cube'
        fem_data = FEMData.read_directory(
            'vtk', data_directory, read_npy=False, save=False).to_first_order()
        effective_normals = fem_data.calculate_surface_normals(
            mode='effective')
        mean_normals = fem_data.calculate_surface_normals(mode='mean')

        nodes = fem_data.nodes.data
        epsilon = 1e-5
        filter_x_low = np.abs(nodes[:, 0] - 0.0) < epsilon
        filter_x_high = np.abs(nodes[:, 0] - 0.1) < epsilon
        filter_y_low = np.abs(nodes[:, 1] - 0.0) < epsilon
        filter_y_high = np.abs(nodes[:, 1] - 0.1) < epsilon
        filter_z_low = np.abs(nodes[:, 2] - 0.0) < epsilon
        filter_z_high = np.abs(nodes[:, 2] - 0.1) < epsilon
        filter_edge = (
            filter_x_low.astype(int) + filter_x_high.astype(int)
            + filter_y_low.astype(int) + filter_y_high.astype(int)
            + filter_z_low.astype(int) + filter_z_high.astype(int)) >= 2

        def assert_almost_equal_broadcast(array, desired):
            np.testing.assert_almost_equal(
                array, np.ones((len(array), len(desired))) * desired)

        assert_almost_equal_broadcast(
            effective_normals[filter_x_low & ~filter_edge], [-1., 0., 0.])
        assert_almost_equal_broadcast(
            mean_normals[filter_x_low & ~filter_edge], [-1., 0., 0.])
        assert_almost_equal_broadcast(
            effective_normals[filter_x_high & ~filter_edge], [1., 0., 0.])
        assert_almost_equal_broadcast(
            mean_normals[filter_x_high & ~filter_edge], [1., 0., 0.])

        assert_almost_equal_broadcast(
            effective_normals[filter_y_low & ~filter_edge], [0., -1., 0.])
        assert_almost_equal_broadcast(
            mean_normals[filter_y_low & ~filter_edge], [0., -1., 0.])
        assert_almost_equal_broadcast(
            effective_normals[filter_y_high & ~filter_edge], [0., 1., 0.])
        assert_almost_equal_broadcast(
            mean_normals[filter_y_high & ~filter_edge], [0., 1., 0.])

        assert_almost_equal_broadcast(
            effective_normals[filter_z_low & ~filter_edge], [0., 0., -1.])
        assert_almost_equal_broadcast(
            mean_normals[filter_z_low & ~filter_edge], [0., 0., -1.])
        assert_almost_equal_broadcast(
            effective_normals[filter_z_high & ~filter_edge], [0., 0., 1.])
        assert_almost_equal_broadcast(
            mean_normals[filter_z_high & ~filter_edge], [0., 0., 1.])

        assert_almost_equal_broadcast(
            effective_normals[
                ~filter_x_low & ~filter_x_high
                & ~filter_y_low & ~filter_y_high
                & ~filter_z_low & ~filter_z_high], [0., 0., 0.])
        assert_almost_equal_broadcast(
            mean_normals[
                ~filter_x_low & ~filter_x_high
                & ~filter_y_low & ~filter_y_high
                & ~filter_z_low & ~filter_z_high], [0., 0., 0.])

    def test_calculate_edge_lengths_tri(self):
        data_directory = 'tests/data/obj/tri'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_edge_lengths = np.array([
            [1.0, np.sqrt(2), 1.0],
            [np.sqrt(2), np.sqrt(2), np.sqrt(2)],
            [np.sqrt(2), 1.0, 1.0],
            [1.0, 1.0, np.sqrt(2)],
        ])
        fem_data.calculate_edge_lengths()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('edge_lengths'),
            desired_edge_lengths, decimal=5)

    def test_calculate_edge_lengths_quad(self):
        data_directory = 'tests/data/obj/quad'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_edge_lengths = np.array([
            [1., 1., 1., 1.],
            [2., np.sqrt(2), 1., 1.],
            [1., np.sqrt(2), 1., np.sqrt(2)],
        ])
        fem_data.calculate_edge_lengths()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('edge_lengths'),
            desired_edge_lengths, decimal=5)

    def test_calculate_angles_tri(self):
        data_directory = 'tests/data/obj/tri'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_angles = np.array([
            [np.pi / 2, np.pi / 4, np.pi / 4],
            [np.pi / 3, np.pi / 3, np.pi / 3],
            [np.pi / 4, np.pi / 4, np.pi / 2],
            [np.pi / 4, np.pi / 2, np.pi / 4],
        ])
        fem_data.calculate_angles()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('angles'),
            desired_angles)

    def test_calculate_angles_quad(self):
        data_directory = 'tests/data/obj/quad'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_angles = np.array([
            [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2],
            [np.pi / 2, np.pi / 4, np.pi * 3 / 4, np.pi / 2],
            [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2],
        ])
        fem_data.calculate_angles()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('angles'),
            desired_angles)

    def test_calculate_jacobians_tri(self):
        data_directory = 'tests/data/obj/tri'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_jacobians = np.array([
            [1],
            [np.sqrt(3)],
            [1],
            [1],
        ])
        fem_data.calculate_jacobians()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('jacobian'),
            desired_jacobians)

    def test_calculate_jacobians_quad(self):
        data_directory = 'tests/data/obj/quad'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        desired_jacobians = np.array([
            0.25,
            0.375,
            0.25 * np.sqrt(2),
        ])
        fem_data.calculate_jacobians()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('jacobian'),
            desired_jacobians)

    def test_extract_direction_feature(self):
        data_directory = 'tests/data/stl/multiple'
        fem_data = FEMData.read_directory(
            'stl', data_directory, read_npy=False, save=False)

        input_vectors = np.array([
            [1., 0., 0.],
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
            [1., 1., 0.],
            [-1., -1., 0.],
        ])
        extracted_vectors = fem_data.extract_direction_feature(input_vectors)
        np.testing.assert_almost_equal(
            extracted_vectors[0], extracted_vectors[1])
        np.testing.assert_almost_equal(
            extracted_vectors[2], extracted_vectors[3])
        np.testing.assert_almost_equal(
            extracted_vectors[4], extracted_vectors[5])
        np.testing.assert_almost_equal(
            extracted_vectors[6], extracted_vectors[7])
        np.testing.assert_almost_equal(
            extracted_vectors[0], extracted_vectors[1])

        n_space = 10
        sp = np.linspace(0., 2 * np.pi, n_space + 1)[:-1]

        # Rotate in XY plane
        input_vectors = np.stack([
            np.sin(sp), np.cos(sp), np.zeros(n_space)], axis=1)
        extracted_vectors = fem_data.extract_direction_feature(input_vectors)
        np.testing.assert_almost_equal(
            extracted_vectors[:n_space // 2]
            - extracted_vectors[n_space // 2:], 0.)

        # Rotate in YZ plane
        input_vectors = np.stack([
            np.zeros(n_space), np.sin(sp), np.cos(sp)], axis=1)
        extracted_vectors = fem_data.extract_direction_feature(input_vectors)
        np.testing.assert_almost_equal(
            extracted_vectors[:n_space // 2]
            - extracted_vectors[n_space // 2:], 0.)

        # Rotate in ZX plane
        input_vectors = np.stack([
            np.sin(sp), np.zeros(n_space), np.cos(sp)], axis=1)
        extracted_vectors = fem_data.extract_direction_feature(input_vectors)
        np.testing.assert_almost_equal(
            extracted_vectors[:n_space // 2]
            - extracted_vectors[n_space // 2:], 0.)

    def test_metric_volume_equal(self):
        file_name = pathlib.Path('tests/data/vtk/mix_hex_hexprism/mesh.vtk')
        fem_data = FEMData.read_files('vtk', [file_name])
        np.testing.assert_almost_equal(
            fem_data.calculate_element_metrics(),
            fem_data.calculate_element_volumes())
        
    def test_integrate_node_attribute_over_surface(self):
        file_name = pathlib.Path('tests/data/fistr/tet/tet.msh')
        fem_data = FEMData.read_files('fistr', [file_name])
        fem_data.nodal_data.set_attribute_data('values', np.array([3,1,4,1,5],np.float32))
        # 123, 124, 134, 235, 245, 345
        sq2 = 2 ** .5
        areas = [1/2, 1/2, 1/2, sq2/2, sq2/2, 1/2]
        vals = [8/3, 5/3, 8/3, 10/3, 7/3, 10/3]
        desired = sum(x*y for x, y in zip(areas, vals))
        actual = fem_data.integrate_node_attribute_over_surface('values')
        np.testing.assert_almost_equal(actual, desired)

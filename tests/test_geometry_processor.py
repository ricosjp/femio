import pathlib
import unittest

import numpy as np

from femio.fem_data import FEMData


RUN_FISTR = True

FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'


class TestGeometryProcessor(unittest.TestCase):

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
        linear_volumes = fem_data.calculate_element_volumes(mode="linear")
        gaussian_volumes = fem_data.calculate_element_volumes(mode="gaussian")
        centroid_volumes = fem_data.calculate_element_volumes(mode="centroid")
        desired_volumes = np.ones((7, 1)) * 8.
        np.testing.assert_almost_equal(linear_volumes, desired_volumes)
        np.testing.assert_almost_equal(gaussian_volumes, desired_volumes)
        np.testing.assert_almost_equal(centroid_volumes, desired_volumes)

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
        linear_areas = fem_data.calculate_element_areas(mode="linear")
        gaussian_areas = fem_data.calculate_element_areas(mode="gaussian")
        centroid_areas = fem_data.calculate_element_areas(mode="centroid")
        np.testing.assert_almost_equal(linear_areas, desired_areas)
        np.testing.assert_almost_equal(gaussian_areas, desired_areas)
        np.testing.assert_almost_equal(centroid_areas, desired_areas)

    def test_calculate_area_quad_gaussian(self):
        data_directory = 'tests/data/obj/quad'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)

        fem_data.nodal_data.reset()
        vx, vy, vz = np.random.random(3)
        theta = np.random.random() * np.pi * 2
        fem_data.rotation(vx, vy, vz, theta)
        vx, vy, vz = np.random.random(3)
        fem_data.translation(vx, vy, vz)

        desired_areas = np.array([
            [1.],
            [1.5],
            [np.sqrt(2)],
        ])
        actual = fem_data.calculate_element_areas(mode="gaussian")
        np.testing.assert_almost_equal(actual, desired_areas)

    def test_calculate_area_polygon(self):
        data_directory = 'tests/data/vtp/polys'
        fem_data = FEMData.read_directory(
            'vtp', data_directory, read_npy=False, save=False)
        desired_areas = np.array([
            [1.25],
            [1.25],
            [1.16],
            [1.25],
        ])
        linear_areas = fem_data.calculate_element_areas(mode="linear")
        centroid_areas = fem_data.calculate_element_areas(mode="centroid")
        np.testing.assert_almost_equal(linear_areas, desired_areas)
        np.testing.assert_almost_equal(centroid_areas, desired_areas)

    def test_calculate_area_nonconvex(self):
        data_directory = 'tests/data/vtp/nonconvex'
        fem_data = FEMData.read_directory(
            'vtp', data_directory, read_npy=False, save=False)
        desired_areas = np.array([
            [3.],
        ])
        linear_areas = fem_data.calculate_element_areas(mode="linear")
        centroid_areas = fem_data.calculate_element_areas(mode="centroid")
        np.testing.assert_almost_equal(linear_areas, desired_areas)
        np.testing.assert_almost_equal(centroid_areas, desired_areas)

    def test_calculate_volumes_hex_gaussian(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/hex_cross', read_npy=False,
            save=False)

        fem_data.nodal_data.reset()
        fem_data.elemental_data.reset()
        vx, vy, vz = np.random.random(3)
        theta = np.random.random() * np.pi * 2
        fem_data.rotation(vx, vy, vz, theta)
        vx, vy, vz = np.random.random(3)
        fem_data.translation(vx, vy, vz)

        desired_volumes = np.full((7, 1), 8.0)
        actual = fem_data.calculate_element_volumes(mode="gaussian")
        np.testing.assert_almost_equal(actual, desired_volumes)

    def test_calculate_volumes_polyhedron(self):
        fem_data = FEMData.read_files(
            'vtk', 'tests/data/vtk/tet_3/mesh.vtk',
        )
        desired = fem_data.calculate_element_volumes()
        fem_data = fem_data.to_polyhedron()
        actual = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(actual, desired)

        fem_data = FEMData.read_files(
            'vtk', 'tests/data/vtk/hex/mesh.vtk',
        )
        desired = fem_data.calculate_element_volumes()
        fem_data = fem_data.to_polyhedron()
        actual = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(actual, desired)

        fem_data = FEMData.read_files(
            'ucd', 'tests/data/ucd/prism/mesh.inp',
        )
        desired = fem_data.calculate_element_volumes()
        fem_data = fem_data.to_polyhedron()
        actual = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(actual, desired)

        fem_data = FEMData.read_files(
            'polyvtk', 'tests/data/vtu/pyramid/pyramid.vtu',
        )
        desired = fem_data.calculate_element_volumes()
        fem_data = fem_data.to_polyhedron()
        actual = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(actual, desired)

        fem_data = FEMData.read_files(
            'polyvtk', 'tests/data/vtu/polyhedron/polyhedron.vtu',
        )
        desired = np.array([[19 / 24], [3 / 2], [1 / 24]])
        actual = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(actual, desired)

    def test_normal_incidence_hex(self):
        fem_data = FEMData.read_files(
            'vtk', ['tests/data/vtk/hex/mesh.vtk'])
        _, inc_facet2cell, normals \
            = fem_data.calculate_normal_incidence_matrix()
        coo = inc_facet2cell.tocoo()
        desired_normals = np.array([
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        all_normals = [
            data * normals[i_facet]
            for data, i_cell, i_facet in zip(coo.data, coo.row, coo.col)]
        np.testing.assert_almost_equal(all_normals, desired_normals)

    def test_normal_incidence_openfoam(self):
        # fem_data = FEMData.read_files(
        #     'vtu', ['tests/data/vtu/complex/mesh.vtu'])
        fem_data = FEMData.read_files(
            'vtu', ['tests/data/vtu/openfoam/internal.vtu'])
        _, inc_facet2cell, normals \
            = fem_data.calculate_normal_incidence_matrix()
        coo = inc_facet2cell.tocoo()
        desired_normals = np.array([
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        all_normals = [
            data * normals[i_facet]
            for data, i_cell, i_facet in zip(coo.data, coo.row, coo.col)]
        np.testing.assert_almost_equal(all_normals, desired_normals)

    def test_relative_incidence_graph_tet1(self):
        fem_data = FEMData.read_files(
            'fistr', ['tests/data/fistr/graph_tet1/mesh.msh'])
        facet_fem_data, inc_facet2cell, normals \
            = fem_data.calculate_normal_incidence_matrix()
        coo = inc_facet2cell.tocoo()
        write_file_name = pathlib.Path(
            'tests/data/ucd/write_graph_tet1_facet/mesh.inp')
        if write_file_name.exists():
            write_file_name.unlink()
        facet_fem_data.write('ucd', write_file_name)
        desired_normals = np.array([
            # Cell 1
            [3**-.5, 3**-.5, 3**-.5],
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            # Cell 2
            [0., -2.**-.5, -2.**-.5],
            [-1., 0., 0.],
            [2.**-.5, 2**-.5, 0.],
            [0., 0., 1.],
            # Cell 3
            [-3**-.5, -3**-.5, -3**-.5],
            [2.**-.5, 2**-.5, 0.],
            [0., -1., 0.],
            [0., 2.**-.5, 2.**-.5],
            # Cell 4
            [0., 0., -1.],
            [0., -1., 0.],
            [-1., 0., 0.],
            [3**-.5, 3**-.5, 3**-.5],
            # Cell 5
            [3**-.5, 3**-.5, -3**-.5],
            [0., 0., 1.],
            [0., -1., 0.],
            [-1., 0., 0.],
        ])
        all_normals = np.array([
            data * normals[i_facet]
            for data, i_cell, i_facet in zip(coo.data, coo.row, coo.col)])
        # raise ValueError(facet_fem_data.elements.data[coo.col], coo.row)
        np.testing.assert_almost_equal(all_normals, desired_normals)

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
        normals = fem_data.calculate_element_normals()
        normals_centroid = fem_data.calculate_element_normals(mode="centroid")
        np.testing.assert_almost_equal(
            fem_data.extract_direction_feature(normals),
            fem_data.extract_direction_feature(desired_normals), decimal=5)
        np.testing.assert_almost_equal(
            fem_data.extract_direction_feature(normals_centroid),
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

    def test_calculate_surface_normals_polyhedron(self):
        data_directory = pathlib.Path('tests/data/vtu/polyhedron_tet')
        fem_data = FEMData.read_directory(
            'polyvtk', data_directory, read_npy=False, save=False)
        normals = fem_data.calculate_surface_normals()
        fem_data.write(
            'polyvtk', 'tests/data/vtu/write_polyhedron_tet/mesh.vtu',
            overwrite=True)
        fem_data.to_surface().write(
            'vtp', 'tests/data/vtp/write_surface_polyhedron_tet/mesh.vtp',
            overwrite=True)
        desired_normals = np.array([
            [0.00000000e+00, -9.61523948e-01, -2.74721128e-01],
            [7.42781385e-01, -5.57085973e-01, -3.71390674e-01],
            [0.00000000e+00, 8.94427200e-01, -4.47213578e-01],
            [-7.42781385e-01, -5.57085973e-01, -3.71390674e-01],
            [9.77802415e-01, 4.04625215e-08, 2.09529086e-01],
            [0.00000000e+00, 9.77802415e-01, 2.09529086e-01],
            [-9.77802415e-01, 4.04625215e-08, 2.09529086e-01],
            [0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
            [6.66666684e-01, -6.66666645e-01, 3.33333342e-01],
            [0.00000000e+00, 8.94427191e-01, 4.47213595e-01],
            [-6.66666684e-01, -6.66666645e-01, 3.33333342e-01],
            [0.00000000e+00, 5.96046431e-08, 1.00000000e+00],
        ])
        np.testing.assert_almost_equal(normals, desired_normals)
        np.testing.assert_almost_equal(
            np.linalg.norm(normals, axis=1), 1.)

    def test_calculate_surface_normals_polyhedron_complex(self):
        data_file = pathlib.Path('tests/data/vtu/complex/mesh.vtu')
        fem_data = FEMData.read_files('polyvtk', data_file)
        normals = fem_data.calculate_surface_normals()
        fem_data.write(
            'polyvtk', 'tests/data/vtu/write_complex/mesh.vtu',
            overwrite=True)
        surface_fem_data = fem_data.to_surface()
        surface_fem_data.write(
            'vtp', 'tests/data/vtp/write_surface_complex/mesh.vtp',
            overwrite=True)
        desired_normals = np.array([
            [-0.57735027, -0.57735027, -0.57735027],
            [0., -0.6, -0.8],
            [0., -0.4472136, -0.89442719],
            [0.40824829, -0.40824829, -0.81649658],
            [-0.57735027, 0.57735027, -0.57735027],
            [0., 0.6, -0.8],
            [0., 0.4472136, -0.89442719],
            [0.40824829, 0.40824829, -0.81649658],
            [0., -1., 0.],
            [0., -1., 0.],
            [0.51449576, -0.85749292, 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0.51449575, 0.85749293, 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [-0.57735027, -0.57735027, 0.57735027],
            [0., -0.6, 0.8],
            [0., -0.64018439, 0.76822128],
            [0.1825742, -0.91287092, 0.3651484],
            [-0.57735027, 0.57735027, 0.57735027],
            [0., 0.6, 0.8],
            [0., 0.6401844, 0.76822128],
            [0.18257418, 0.91287094, 0.36514835],
            [0., 0., 1.],
            [0., 0., 1.],
            [0.4472136, 0., 0.89442719],
            [0.4472136, 0., 0.89442719],
        ])
        np.testing.assert_almost_equal(normals, desired_normals)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_almost_equal(norms[norms > .5], 1.)
        np.testing.assert_almost_equal(norms[norms <= .5], 0.)

        desired_surfaces = {
            'quad': np.array([
                [1, 5, 6, 2],
                [1, 19, 23, 5],
                [2, 6, 7, 3],
                [2, 3, 10, 9],
                [3, 7, 8, 4],
                [3, 4, 11, 10],
                [6, 12, 13, 7],
                [7, 13, 14, 8],
                [9, 10, 21, 20],
                [10, 11, 22, 21],
                [11, 17, 29, 22],
                [12, 24, 25, 13],
                [13, 25, 26, 14],
                [18, 14, 26, 30],
                [17, 18, 30, 29],
                [19, 20, 24, 23],
                [21, 22, 29, 27],
                [28, 30, 26, 25],
                [27, 29, 30, 28]]),
            'polygon': np.array([
                np.array([1, 2, 9, 20, 19]),
                np.array([5, 23, 24, 12, 6]),
                np.array([4, 8, 14, 18, 17, 11]),
                np.array([20, 21, 27, 28, 25, 24])], dtype=object)
        }
        for k in desired_surfaces.keys():
            for actual_e, desired_e in zip(
                    surface_fem_data.elements[k].data, desired_surfaces[k]):
                np.testing.assert_array_equal(actual_e, desired_e)

    def test_calculate_elementl_normals_polygon(self):
        data_directory = pathlib.Path('tests/data/vtp/polys')
        fem_data = FEMData.read_directory(
            'vtp', data_directory, read_npy=False, save=False)
        normals = fem_data.calculate_element_normals()
        desired_normals = np.array([
            [0., 0., -1.],
            [0., 0., 1.],
            [0., -1., 0.],
            [1., 0., 0.],
        ])
        np.testing.assert_almost_equal(normals, desired_normals)

    def test_calculate_elementl_normals_curved_polygon(self):
        data_directory = pathlib.Path('tests/data/vtp/curved')
        fem_data = FEMData.read_directory(
            'vtp', data_directory, read_npy=False, save=False)
        normals = fem_data.calculate_element_normals()

        desired_normals = np.array([
            [0., 0., 1.],
        ])
        np.testing.assert_almost_equal(normals, desired_normals, decimal=2)

        normal_norms = np.linalg.norm(normals, axis=1)
        desired_normal_norms = np.array([1.])
        np.testing.assert_almost_equal(normal_norms, desired_normal_norms)

    def test_calculate_elemental_normals_mixed_polygon(self):
        data_directory = pathlib.Path('tests/data/vtp/closed')
        fem_data = FEMData.read_directory(
            'vtp', data_directory, read_npy=False, save=False)
        normals = fem_data.calculate_element_normals()
        fem_data.write(
            'vtp', 'tests/data/vtp/write_closed_normal/mesh.vtp',
            overwrite=True)
        desired_normals = np.array([
            [0., 0., -1.],
            [0., -1., 0.],
            [0., -1., 0.],
            [1., 0., 0.],
            [1., 0., -0.],
            [0.70710678, 0.70710678, -0.],
            [0.70710678, 0.70710678, -0.],
            [-0.70710678, 0.70710678, 0.],
            [-1., 0., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ])
        np.testing.assert_almost_equal(normals, desired_normals)

    def test_calculate_elemental_areas_mixed_polygon(self):
        data_directory = pathlib.Path('tests/data/vtp/closed')
        fem_data = FEMData.read_directory(
            'vtp', data_directory, read_npy=False, save=False)
        areas = fem_data.calculate_element_areas()
        desired_areas = np.array([
            [1.25],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
            [0.35355338],
            [0.35355338],
            [0.70710677],
            [1.],
            [1.],
            [0.25],
        ])
        np.testing.assert_almost_equal(areas, desired_areas)

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

    def test_translation_and_rotation(self):
        data_directory = 'tests/data/obj/tri'
        fem_data = FEMData.read_directory(
            'obj', data_directory, read_npy=False, save=False)
        fem_data.nodal_data.reset()

        fem_data.rotation(1, 1, 1, np.pi / 2)
        a, b, c = 1 / 3, (1 + 3**.5) / 3, (1 - 3**.5) / 3
        desired = np.array([0, 0, 0, a, b, c, c, a, b, b, c, a]).reshape(4, 3)
        np.testing.assert_almost_equal(fem_data.nodes.data, desired)

        fem_data.translation(1, 2, 3)
        desired = np.array([1, 2, 3, a + 1, b + 2, c + 3, c + 1, a + 2,
                            b + 3, b + 1, c + 2, a + 3]).reshape(4, 3)
        np.testing.assert_almost_equal(fem_data.nodes.data, desired)

        vx, vy, vz = np.random.random(3)
        theta = np.random.random() * np.pi * 2
        fem_data.rotation(vx, vy, vz, theta)
        vx, vy, vz = np.random.random(3)
        fem_data.translation(vx, vy, vz)

        X, Y, Z = fem_data.nodes.data.T
        dist = np.empty((4, 4))
        for i in range(4):
            for j in range(4):
                dist[i, j] = (X[i] - X[j])**2 + \
                    (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2
        desired = np.array([0, 1, 1, 1, 1, 0, 2, 2, 1, 2, 0,
                            2, 1, 2, 2, 0], np.float64).reshape(4, 4)
        np.testing.assert_almost_equal(dist, desired)

    def test_integrate_node_attribute_over_surface(self):
        file_name = pathlib.Path('tests/data/fistr/tet/tet.msh')
        fem_data = FEMData.read_files('fistr', [file_name])
        fem_data.nodal_data.set_attribute_data(
            'values', np.array([3, 1, 4, 1, 5], np.float32))
        # 123, 124, 134, 235, 245, 345
        sq2 = 2 ** .5
        areas = [1 / 2, 1 / 2, 1 / 2, sq2 / 2, sq2 / 2, 1 / 2]
        vals = [8 / 3, 5 / 3, 8 / 3, 10 / 3, 7 / 3, 10 / 3]
        desired = sum(x * y for x, y in zip(areas, vals))
        actual = fem_data.integrate_node_attribute_over_surface('values')
        np.testing.assert_almost_equal(actual, desired)

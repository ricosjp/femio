import pathlib
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestPolyVTK(unittest.TestCase):

    def test_read_vtk_polyhedron(self):
        file_name = pathlib.Path('tests/data/vtu/polyhedron/polyhedron.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])

        desired_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 3.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [0, 1, 3, 5, 8, 10, 11, 9],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12],
        ], dtype=object)
        for ae, de in zip(fem_data.elements.data, desired_elements):
            np.testing.assert_almost_equal(ae, np.array(de) + 1)

        desired_u = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('U'), desired_u)

        desired_x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 30.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('X'), desired_x)

        desired_nodal_adj = np.array([
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ])
        np.testing.assert_array_equal(
            fem_data.calculate_adjacency_matrix_node().toarray().astype(int),
            desired_nodal_adj)

        desired_elemental_adj = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
        ])
        np.testing.assert_array_equal(
            fem_data.calculate_adjacency_matrix_element().toarray()
            .astype(int),
            desired_elemental_adj)

    def test_read_vtk_mix_poly(self):
        file_name = pathlib.Path('tests/data/vtu/mix_poly/mix_poly.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])

        desired_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 3.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [0, 1, 3, 5, 8, 10, 11, 9],
            [1, 2, 4, 3, 5, 6, 7, 8],
            [9, 10, 11, 12],
        ], dtype=object)
        for ae, de in zip(fem_data.elements.data, desired_elements):
            np.testing.assert_almost_equal(ae, np.array(de) + 1)

        desired_u = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('U'), desired_u)

        desired_x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 30.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('X'), desired_x)

        desired_nodal_adj = np.array([
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ])
        np.testing.assert_array_equal(
            fem_data.calculate_adjacency_matrix_node().toarray().astype(int),
            desired_nodal_adj)

        desired_elemental_adj = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
        ])
        np.testing.assert_array_equal(
            fem_data.calculate_adjacency_matrix_element().toarray()
            .astype(int),
            desired_elemental_adj)

    def test_read_vtk_poly_pyramid(self):
        file_name = pathlib.Path('tests/data/vtu/poly_pyramid/mesh.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])

        desired_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 3.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_nodes)

        desired_elements = np.array([
            [0, 1, 3, 4, 5, 7, 8, 6],
            [1, 3, 5, 4, 2],
            [6, 7, 8, 9],
        ], dtype=object)
        for ae, de in zip(fem_data.elements.data, desired_elements):
            np.testing.assert_almost_equal(ae, np.array(de) + 1)

        desired_u = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('U'), desired_u)

        desired_x = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 30.0],
        ])
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('X'), desired_x)

    def test_read_vtk_pyramid(self):
        file_name = pathlib.Path('tests/data/vtu/pyramid/pyramid.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])
        vtk_fem_data = FEMData.read_files('vtk', [file_name])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, vtk_fem_data.nodes.data)

        for ae, de in zip(
                fem_data.elements.data[[3, 1, 2, 0]],
                vtk_fem_data.elements.data):
            np.testing.assert_almost_equal(ae, np.array(de))

        volumes = fem_data.calculate_element_volumes(mode="linear")
        # fem_data.write('polyvtk', 'tmp.vtu', overwrite=True)
        vtk_fem_data.calculate_element_volumes(mode="linear")
        # vtk_fem_data.write('vtk', 'tmp.vtk', overwrite=True)
        np.testing.assert_almost_equal(
            volumes[[3, 1, 2, 0]],
            vtk_fem_data.calculate_element_volumes(mode="linear"))

    def test_read_vtk_voxel(self):
        file_name = pathlib.Path('tests/data/vtu/voxel/mesh.vtu')
        fem_data = FEMData.read_files('polyvtk', [file_name])

        volumes = fem_data.calculate_element_volumes(mode="linear")
        np.testing.assert_almost_equal(volumes, .5)
        write_file_name = 'tests/data/vtu/write_voxel/mesh.vtu'
        fem_data.write(
            'polyvtk', write_file_name, overwrite=True)
        written_fem_data = FEMData.read_files('polyvtk', write_file_name)
        np.testing.assert_array_equal(
            written_fem_data.elements.data - 1,
            np.array([
                [3, 9, 13, 12, 7, 11, 15, 14],  # Original hex comes first
                [0, 1, 3, 2, 4, 5, 7, 6],
                [1, 8, 9, 3, 5, 10, 11, 7],
            ])
        )

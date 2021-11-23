import unittest

import numpy as np

from femio.fem_data import FEMData
from femio.util import brick_generator


WRITE_TEST_DATA = True


class TestBrickGenerator(unittest.TestCase):

    def test_generate_brick_tri(self):
        n_x_element = 4
        n_y_element = 10
        x_length = .5
        y_length = 2.

        fem_data = brick_generator.generate_brick(
            'tri', n_x_element, n_y_element,
            x_length=x_length, y_length=y_length)
        np.testing.assert_almost_equal(
            np.max(fem_data.nodes.data, axis=0), [x_length, y_length, 0.])
        np.testing.assert_almost_equal(
            np.min(fem_data.nodes.data, axis=0), [0., 0., 0.])
        self.assertEqual(len(fem_data.elements), n_x_element * n_y_element * 2)

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/brick_tri/mesh.inp', overwrite=True)
        areas = fem_data.calculate_element_areas()
        np.testing.assert_almost_equal(
            areas, (x_length / n_x_element) * (y_length / n_y_element) / 2)
        ref_fem_data = FEMData.read_directory(
            'ucd', 'tests/data/ucd/brick_tri', read_npy=False, save=False)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, ref_fem_data.nodes.data)
        np.testing.assert_array_equal(
            fem_data.elements.data, ref_fem_data.elements.data)

    def test_generate_brick_quad(self):
        n_x_element = 4
        n_y_element = 20
        x_length = 1.
        y_length = 4.

        fem_data = brick_generator.generate_brick(
            'quad', n_x_element, n_y_element,
            x_length=x_length, y_length=y_length)
        np.testing.assert_almost_equal(
            np.max(fem_data.nodes.data, axis=0), [x_length, y_length, 0.])
        np.testing.assert_almost_equal(
            np.min(fem_data.nodes.data, axis=0), [0., 0., 0.])
        self.assertEqual(len(fem_data.elements), n_x_element * n_y_element)

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/brick_quad/mesh.inp', overwrite=True)
        areas = fem_data.calculate_element_areas()
        np.testing.assert_almost_equal(
            areas, (x_length / n_x_element) * (y_length / n_y_element))
        ref_fem_data = FEMData.read_directory(
            'ucd', 'tests/data/ucd/brick_quad', read_npy=False, save=False)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, ref_fem_data.nodes.data)
        np.testing.assert_array_equal(
            fem_data.elements.data, ref_fem_data.elements.data)

    def test_generate_brick_tet(self):
        n_x_element = 10
        n_y_element = 4
        n_z_element = 3
        x_length = 20.
        y_length = 2.
        z_length = 5.

        fem_data = brick_generator.generate_brick(
            'tet', n_x_element, n_y_element, n_z_element,
            x_length=x_length, y_length=y_length, z_length=z_length)
        np.testing.assert_almost_equal(
            np.max(fem_data.nodes.data, axis=0),
            [x_length, y_length, z_length])
        np.testing.assert_almost_equal(
            np.min(fem_data.nodes.data, axis=0), [0., 0., 0.])
        self.assertEqual(
            len(fem_data.elements),
            n_x_element * n_y_element * n_z_element * 6)

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/brick_tet/mesh.inp', overwrite=True)
        volumes = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(
            volumes,
            (x_length / n_x_element) * (y_length / n_y_element)
            * (z_length / n_z_element) * np.array(
                [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
                * n_x_element * n_y_element * n_z_element)[..., None])
        ref_fem_data = FEMData.read_directory(
            'ucd', 'tests/data/ucd/brick_tet', read_npy=False, save=False)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, ref_fem_data.nodes.data)
        np.testing.assert_array_equal(
            fem_data.elements.data, ref_fem_data.elements.data)

    def test_generate_brick_tet_face_consistent(self):
        n_x_element = 2
        n_y_element = 3
        n_z_element = 4
        x_length = 1.
        y_length = 1.
        z_length = 1.

        fem_data = brick_generator.generate_brick(
            'tet', n_x_element, n_y_element, n_z_element,
            x_length=x_length, y_length=y_length, z_length=z_length)
        surface_fem_data = fem_data.to_surface()

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/write_brick_tet/mesh.inp',
                overwrite=True)
            surface_fem_data.write(
                'ucd', 'tests/data/ucd/write_surface_tri/mesh.inp',
                overwrite=True)

        self.assertEqual(
            len(surface_fem_data.elements), (2 * 3 + 3 * 4 + 4 * 2) * 2 * 2)
        volumes = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(
            volumes,
            (x_length / n_x_element) * (y_length / n_y_element)
            * (z_length / n_z_element) * np.array(
                [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
                * n_x_element * n_y_element * n_z_element)[..., None])

    def test_generate_brick_hex(self):
        n_x_element = 3
        n_y_element = 4
        n_z_element = 10
        x_length = .1
        y_length = 2.
        z_length = 5.

        fem_data = brick_generator.generate_brick(
            'hex', n_x_element, n_y_element, n_z_element,
            x_length=x_length, y_length=y_length, z_length=z_length)
        np.testing.assert_almost_equal(
            np.max(fem_data.nodes.data, axis=0),
            [x_length, y_length, z_length])
        np.testing.assert_almost_equal(
            np.min(fem_data.nodes.data, axis=0), [0., 0., 0.])
        self.assertEqual(
            len(fem_data.elements),
            n_x_element * n_y_element * n_z_element)

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/brick_hex/mesh.inp', overwrite=True)
        volumes = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(
            volumes,
            (x_length / n_x_element) * (y_length / n_y_element)
            * (z_length / n_z_element))
        ref_fem_data = FEMData.read_directory(
            'ucd', 'tests/data/ucd/brick_hex', read_npy=False, save=False)
        np.testing.assert_almost_equal(
            fem_data.nodes.data, ref_fem_data.nodes.data)
        np.testing.assert_array_equal(
            fem_data.elements.data, ref_fem_data.elements.data)

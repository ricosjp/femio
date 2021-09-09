import pathlib
import shutil
import subprocess
import unittest

import numpy as np

from femio.fem_data import FEMData
from femio.util import random_generator


WRITE_TEST_DATA = True


class TestBrickGenerator(unittest.TestCase):

    def test_generate_random_tri(self):
        n_point = 100
        x_length = .5
        y_length = 2.

        fem_data = random_generator.generate_random_mesh(
            'tri', n_point,
            x_length=x_length, y_length=y_length, noise_scale=0.)
        self.assertEqual(len(fem_data.nodes), n_point)
        np.testing.assert_almost_equal(
            np.max(fem_data.nodes.data, axis=0), [.5, 2., 0.], decimal=2)
        np.testing.assert_almost_equal(
            np.min(fem_data.nodes.data, axis=0), [0., 0., 0.], decimal=2)

        areas = fem_data.calculate_element_areas()
        np.testing.assert_almost_equal(
            np.sum(areas), x_length * y_length, decimal=1)

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/random_tri/mesh.inp', overwrite=True)

    def test_generate_random_tri_filter(self):
        n_point = 200
        x_length = .5
        y_length = 2.

        fem_data = random_generator.generate_random_mesh(
            'tri', n_point,
            x_length=x_length, y_length=y_length, quality_threshold=.1,
            noise_scale=1.)
        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/random_tri_filter/mesh.inp',
                overwrite=True)

    def test_generate_random_tet(self):
        n_point = 10 * 2 * 5 * 4 * 4 * 4
        x_length = 10.
        y_length = 2.
        z_length = 5.

        fem_data = random_generator.generate_random_mesh(
            'tet', n_point,
            x_length=x_length, y_length=y_length, z_length=z_length,
            noise_scale=0.)
        self.assertEqual(len(fem_data.nodes), n_point)
        np.testing.assert_almost_equal(
            np.max(fem_data.nodes.data, axis=0),
            [x_length, y_length, z_length], decimal=2)
        np.testing.assert_almost_equal(
            np.min(fem_data.nodes.data, axis=0), [0., 0., 0.], decimal=2)

        volumes = fem_data.calculate_element_volumes()
        np.testing.assert_almost_equal(
            np.sum(volumes), x_length * y_length * z_length, decimal=1)

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/random_tet/mesh.inp', overwrite=True)

    def test_generate_random_tet_strip(self):
        n_point = 10 * 2 * 5 * 4 * 4 * 4
        x_length = 10.
        y_length = 2.
        z_length = 5.

        fem_data = random_generator.generate_random_mesh(
            'tet', n_point,
            x_length=x_length, y_length=y_length, z_length=z_length,
            noise_scale=1., strip_epsilon=.01)

        fistr_write_directory = pathlib.Path(
            'tests/data/fistr/write_random_tet_strip')
        if fistr_write_directory.exists():
            shutil.rmtree(fistr_write_directory)
        fem_data.add_static_material()
        fem_data.constraints.update_data(
            fem_data.nodes.ids,
            {'boundary': np.zeros((len(fem_data.nodes), 3))})
        fem_data.constraints['boundary'].data[-1] = np.ones(3) * 1.e-3

        fem_data.write('fistr', fistr_write_directory / 'mesh')

        subprocess.check_output(
            "echo hi; fistr1 2>&1; echo hi", cwd=fistr_write_directory,
            shell=True)
        written_fem_data_with_res = FEMData.read_directory(
            'fistr', fistr_write_directory, read_npy=False, save=False)

        np.testing.assert_almost_equal(
            written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
            fem_data.constraints['boundary'].data)

        if WRITE_TEST_DATA:
            fem_data.write(
                'ucd', 'tests/data/ucd/random_tet_strip/mesh.inp',
                overwrite=True)

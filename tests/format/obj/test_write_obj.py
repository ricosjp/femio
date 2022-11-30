import os
import unittest

import numpy as np

from femio.fem_data import FEMData


class TestWriteOBJ(unittest.TestCase):

    def test_write_obj(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/thermal/hex.msh',
            'tests/data/fistr/thermal/hex.cnt',
            'tests/data/fistr/thermal/hex.res.0.1'])
        write_file_name = 'tests/data/obj/write/mesh.obj'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('obj', write_file_name)

        obj_fem_data = FEMData.read_files(
            'obj', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data,
            obj_fem_data.nodes.data)

    def test_write_obj_mixture_solid(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid_simple', read_npy=False)
        write_file_name = 'tests/data/obj/write_mixture_solid_simple/mesh.obj'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('obj', write_file_name)

        obj_fem_data = FEMData.read_files(
            'obj', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data,
            obj_fem_data.nodes.data)

    def test_write_obj_mixture_shell(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_shell', read_npy=False)
        write_file_name = 'tests/data/obj/write_mixture_shell/mesh.obj'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('obj', write_file_name)

        obj_fem_data = FEMData.read_files(
            'obj', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data,
            obj_fem_data.nodes.data)

    def test_write_obj_polygon(self):
        fem_data = FEMData.read_files(
            'vtu', 'tests/data/vtu/polyhedron/polyhedron.vtu')
        fem_data = fem_data.to_surface()
        write_file_name = 'tests/data/obj/write/mesh.obj'
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        fem_data.write('obj', write_file_name)

        obj_fem_data = FEMData.read_files(
            'obj', write_file_name)
        np.testing.assert_almost_equal(
            fem_data.nodes.data,
            obj_fem_data.nodes.data)
        for i in range(len(fem_data.elements.data)):
            np.testing.assert_array_equal(
                fem_data.elements.data[i],
                obj_fem_data.elements.data[i],
            )

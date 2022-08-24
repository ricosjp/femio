import unittest
import numpy as np
import femio
from femio.mesh_compressor import MeshCompressor, calc_centers, merge_elements


class TestCase(unittest.TestCase):
    def test_calc_centers(self):
        fem_data = femio.read_files(
            'polyvtk', 'tests/data/vtu/brick/brick_1.vtu')
        csr = fem_data.face_data_csr()
        node_pos = fem_data.nodes.data
        desired = np.array([
            [0.7, 0.76, 0.48],
            [0.64, 0.32, 0.7],
            [0.64, 0.32, 0.2],
            [8 / 30, 0.65, 29 / 60],
            [0.24, 0.2, 0.48]
        ])
        actual = calc_centers(csr, node_pos)
        np.testing.assert_array_equal(desired, actual)

    def test_merge_elements(self):
        fem_data = femio.read_files(
            'polyvtk', 'tests/data/vtu/polyhedron/polyhedron.vtu')
        csr = fem_data.face_data_csr()
        node_pos = fem_data.nodes.data
        elem_conv = np.arange(len(fem_data.elements.data), dtype=np.int32)
        csr = merge_elements(csr, node_pos, elem_conv, 2)
        indptr, dat = csr
        poly1 = dat[indptr[0]:indptr[1]]
        poly2 = dat[indptr[1]:indptr[2]]
        desired1 = np.array([
            4, 3, 9, 11, 10, 3, 9, 10, 12, 3, 10, 11, 12, 3, 9, 12, 11
        ])
        desired2 = np.array(
            [10, 5, 0, 1, 5, 10, 9, 5, 0, 9, 11, 8, 3, 4, 5, 8, 11,
             10, 3, 9, 10, 11, 3, 0, 3, 1, 4, 1, 2, 6, 5, 4, 2, 4,
             7, 6, 4, 5, 6, 7, 8, 4, 3, 8, 7, 4, 4, 1, 3, 4, 2
             ])
        np.testing.assert_array_equal(desired1, poly1)
        np.testing.assert_array_equal(desired2, poly2)

    def test_brick(self):
        fem_data = femio.read_files(
            'polyvtk', 'tests/data/vtu/brick/brick.vtu')
        compressor = MeshCompressor(fem_data=fem_data)
        compressor.compress(elem_num=4, cos_thresh=0.9, dist_thresh=0)
        new_fem_data = compressor.output_fem_data
        fem_data_1 = femio.read_files(
            'polyvtk', 'tests/data/vtu/brick/brick_1.vtu')
        np.testing.assert_array_equal(
            new_fem_data.nodes.data,
            fem_data_1.nodes.data)
        for P, Q in zip(new_fem_data.elements.data,
                        fem_data_1.elements.data):
            np.testing.assert_array_equal(P, Q)
        for P, Q in zip(new_fem_data.elemental_data['face'].data,
                        fem_data_1.elemental_data['face'].data):
            np.testing.assert_array_equal(P, Q)

    def test_merge_vertices(self):
        fem_data = femio.read_files(
            'polyvtk', 'tests/data/vtu/brick/brick_1.vtu')
        compressor = MeshCompressor(fem_data=fem_data)
        compressor.compress(elem_num=5, cos_thresh=0.9, dist_thresh=0.3)
        new_fem_data = compressor.output_fem_data
        fem_data_1 = femio.read_files(
            'polyvtk', 'tests/data/vtu/brick/brick_2.vtu')
        np.testing.assert_array_equal(
            new_fem_data.nodes.data,
            fem_data_1.nodes.data)
        for P, Q in zip(new_fem_data.elements.data,
                        fem_data_1.elements.data):
            np.testing.assert_array_equal(P, Q)
        for P, Q in zip(new_fem_data.elemental_data['face'].data,
                        fem_data_1.elemental_data['face'].data):
            np.testing.assert_array_equal(P, Q)

    def test_run_without_err(self):
        import itertools
        es = [1, 2, 3, 4, 5, 10, 100, 200]
        cs = [1.2, 1.0, 0.8, 0.0]
        ds = [0.0, 0.2, 0.3, 0.5, 1.1]
        fem_data = femio.read_files(
            'polyvtk', 'tests/data/vtu/brick/brick.vtu')

        for e, c, d in itertools.product(es, cs, ds):
            compressor = MeshCompressor(fem_data=fem_data)
            compressor.compress(elem_num=e, cos_thresh=c, dist_thresh=d)

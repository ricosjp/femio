import unittest

import numpy as np
from scipy import sparse as sp

import femio.functions as functions


class TestFunctions(unittest.TestCase):

    def test_normalize(self):
        array = np.random.rand(10, 3)
        normed = functions.normalize(array)
        for n in normed:
            np.testing.assert_almost_equal(np.linalg.norm(n), 1.)

    def test_align_nnz(self):
        row = np.array([0, 2])
        col = np.array([0, 1])
        data = np.array([10, 20])
        csr = sp.csr_matrix((data, (row, col)), shape=(3, 3))

        ref_row = np.array([0, 1, 2, 2])
        ref_col = np.array([0, 1, 1, 2])
        ref_data = np.array([1, 2, 0, 8])
        ref_csr = sp.csr_matrix((ref_data, (ref_row, ref_col)), shape=(3, 3))

        aligned_csr = functions.align_nnz(csr, ref_csr)

        desired_row = np.array([0, 1, 2, 2])
        desired_col = np.array([0, 1, 1, 2])
        desired_data = np.array([10, 0, 20, 0])
        desired_csr = sp.csr_matrix(
            (desired_data, (desired_row, desired_col)), shape=(3, 3))
        np.testing.assert_array_equal(aligned_csr.data, desired_csr.data)
        np.testing.assert_array_equal(aligned_csr.indices, desired_csr.indices)
        np.testing.assert_array_equal(aligned_csr.indptr, desired_csr.indptr)

import os
from pathlib import Path
import shutil
import unittest

import numpy as np
import pandas as pd
from scipy import sparse as sp

from femio.fem_data import FEMData
import femio.functions as functions


RUN_FISTR = True


class TestFunctions(unittest.TestCase):

    def test_normalize(self):
        array = np.random.rand(10, 3)
        normed = functions.normalize(array)
        for n in normed:
            np.testing.assert_almost_equal(np.linalg.norm(n), 1.)

    def test_align_nnz(self):
        row = np.array([0, 1, 2])
        col = np.array([0, 2, 1])
        data = np.array([10, 20, 30])
        csr = sp.csr_matrix((data, (row, col)), shape=(3, 3))

        ref_row = np.array([0, 1, 2, 2])
        ref_col = np.array([0, 1, 1, 2])
        ref_data = np.array([1, 2, 0, 8])
        ref_csr = sp.csr_matrix((ref_data, (ref_row, ref_col)), shape=(3, 3))

        aligned_csr, aligned_ref_csr = functions.align_nnz([csr, ref_csr])

        desired_row = np.array([0, 1, 1, 2, 2])
        desired_col = np.array([0, 1, 2, 1, 2])
        desired_data = np.array([10, 0, 20, 30, 0])
        desired_csr = sp.csr_matrix(
            (desired_data, (desired_row, desired_col)), shape=(3, 3))
        np.testing.assert_array_equal(aligned_csr.data, desired_csr.data)
        np.testing.assert_array_equal(aligned_csr.indices, desired_csr.indices)
        np.testing.assert_array_equal(aligned_csr.indptr, desired_csr.indptr)

        desired_ref_row = np.array([0, 1, 1, 2, 2])
        desired_ref_col = np.array([0, 1, 2, 1, 2])
        desired_ref_data = np.array([1, 2, 0, 0, 8])
        desired_ref_csr = sp.csr_matrix(
            (desired_ref_data, (desired_ref_row, desired_ref_col)),
            shape=(3, 3))
        np.testing.assert_array_equal(
            aligned_ref_csr.data, desired_ref_csr.data)
        np.testing.assert_array_equal(
            aligned_ref_csr.indices, desired_ref_csr.indices)
        np.testing.assert_array_equal(
            aligned_ref_csr.indptr, desired_ref_csr.indptr)

    def test_array2symmetric_matrix(self):
        in_array = np.array([
            [1.11, 1.22, 1.33, 1.12, 1.23, 1.31],
            [2.11, 2.22, 2.33, 2.12, 2.23, 2.31],
            [3.11, 3.22, 3.33, 3.12, 3.23, 3.31],
            [4.11, 4.22, 4.33, 4.12, 4.23, 4.31],
        ])
        desired_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        out_array = functions.convert_array2symmetric_matrix(in_array)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_array2symmetric_matrix_engineering(self):
        in_array = np.array([
            [1.11, 1.22, 1.33, 1.12 * 2, 1.23 * 2, 1.31 * 2],
            [2.11, 2.22, 2.33, 2.12 * 2, 2.23 * 2, 2.31 * 2],
            [3.11, 3.22, 3.33, 3.12 * 2, 3.23 * 2, 3.31 * 2],
            [4.11, 4.22, 4.33, 4.12 * 2, 4.23 * 2, 4.31 * 2],
        ])
        desired_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        out_array = functions.convert_array2symmetric_matrix(
                in_array, from_engineering=True)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_array2symmetric_matrix_order(self):
        in_array = np.array([
            [1.11, 1.12, 1.31, 1.22, 1.23, 1.33],
            [2.11, 2.12, 2.31, 2.22, 2.23, 2.33],
            [3.11, 3.12, 3.31, 3.22, 3.23, 3.33],
            [4.11, 4.12, 4.31, 4.22, 4.23, 4.33],
        ])
        desired_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        out_array = functions.convert_array2symmetric_matrix(
            in_array, order=[0, 3, 5, 1, 4, 2])
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_symmetric_matrix2array(self):
        in_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        desired_array = np.array([
            [1.11, 1.22, 1.33, 1.12, 1.23, 1.31],
            [2.11, 2.22, 2.33, 2.12, 2.23, 2.31],
            [3.11, 3.22, 3.33, 3.12, 3.23, 3.31],
            [4.11, 4.22, 4.33, 4.12, 4.23, 4.31],
        ])
        out_array = functions.convert_symmetric_matrix2array(in_array)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_symmetric_matrix2array_to_engineering(self):
        in_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        desired_array = np.array([
            [1.11, 1.22, 1.33, 1.12 * 2, 1.23 * 2, 1.31 * 2],
            [2.11, 2.22, 2.33, 2.12 * 2, 2.23 * 2, 2.31 * 2],
            [3.11, 3.22, 3.33, 3.12 * 2, 3.23 * 2, 3.31 * 2],
            [4.11, 4.22, 4.33, 4.12 * 2, 4.23 * 2, 4.31 * 2],
        ])
        out_array = functions.convert_symmetric_matrix2array(
            in_array, to_engineering=True)
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_symmetric_matrix2array_order(self):
        in_array = np.array([
            [
                [1.11, 1.12, 1.31],
                [1.12, 1.22, 1.23],
                [1.31, 1.23, 1.33],
            ],
            [
                [2.11, 2.12, 2.31],
                [2.12, 2.22, 2.23],
                [2.31, 2.23, 2.33],
            ],
            [
                [3.11, 3.12, 3.31],
                [3.12, 3.22, 3.23],
                [3.31, 3.23, 3.33],
            ],
            [
                [4.11, 4.12, 4.31],
                [4.12, 4.22, 4.23],
                [4.31, 4.23, 4.33],
            ],
        ])
        desired_array = np.array([
            [1.11, 1.12, 1.31, 1.22, 1.23, 1.33],
            [2.11, 2.12, 2.31, 2.22, 2.23, 2.33],
            [3.11, 3.12, 3.31, 3.22, 3.23, 3.33],
            [4.11, 4.12, 4.31, 4.22, 4.23, 4.33],
        ])
        out_array = functions.convert_symmetric_matrix2array(
                in_array, order=[0, 3, 5, 1, 4, 2])
        np.testing.assert_almost_equal(out_array, desired_array)

    def test_calculate_principal_components(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/lte_direction',
            read_npy=False, save=False)
        lte, directions, lte_vec = functions.calculate_principal_components(
            fem_data.elemental_data.get_attribute_data('lte_full'),
            from_engineering=True)

        np.testing.assert_almost_equal(
            lte[0, :],
            np.array([.01, .001, .0001]))
        np.testing.assert_almost_equal(
            directions[0],
            np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.]))
        np.testing.assert_almost_equal(
            lte_vec[0],
            np.array([.01, 0., 0., 0., .001, 0., 0., 0., .0001]))

        np.testing.assert_almost_equal(
            lte[1, :],
            np.array([0.01, 0.001, 0.0001]))
        np.testing.assert_almost_equal(
            directions[1],
            np.array([0., 0., 1., 0., 1., 0., -1., 0., 0.]))
        # raise ValueError(lte[1, :], directions[1], lte_vec[1])
        np.testing.assert_almost_equal(
            lte_vec[1],
            np.array([0., 0., .01, 0., .001, 0., -.0001, 0., 0.]))

        np.testing.assert_almost_equal(
            lte[2, :],
            np.array([0.003, 0.002, 0.001]))
        np.testing.assert_almost_equal(
            directions[2],
            np.array([
                0., 0., 1.,
                -1 / 2**.5, 1 / 2**.5, 0.,
                -1 / 2**.5, -1 / 2**.5, 0.]))
        np.testing.assert_almost_equal(
            lte_vec[2],
            np.array([
                0., 0., .003,
                -.002 / 2**.5, .002 / 2**.5, 0.,
                -.001 / 2**.5, -.001 / 2**.5, 0.]))

    def test_add_principal_vectors(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/lte_direction',
            read_npy=False, save=False)
        _, _, lte_vec = functions.calculate_principal_components(
            fem_data.elemental_data.get_attribute_data('lte_full'),
            from_engineering=True)
        fem_data.add_principal_vectors('lte_full')
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('principal_lte_full_1'),
            lte_vec[:, :3])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('principal_lte_full_2'),
            lte_vec[:, 3:6])
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('principal_lte_full_3'),
            lte_vec[:, 6:])

    def test_invert_strain_xx(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/invert_strain_xx',
            read_npy=False, save=False)

        write_dir_name = Path('tests/data/fistr/write_invert_strain_xx')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)

        fem_data.elemental_data.pop(
            'linear_thermal_expansion_coefficient', None)
        fem_data.elemental_data.pop(
            'ORIENTATION', None)
        fem_data.elemental_data.pop(
            'INITIAL_TEMPERATURE', None)
        fem_data.elemental_data.pop(
            'CNT_TEMPERATURE', None)
        original_nodes = fem_data.nodes.data
        fem_data.nodes.data = fem_data.nodes.data \
            + fem_data.nodal_data.get_attribute_data('DISPLACEMENT')
        fem_data.settings['solution_type'] = 'EPS2DISP'
        fem_data.write('fistr', write_dir_name / 'mesh')

        istrain_file_name = write_dir_name / 'istrain.dat'
        with open(istrain_file_name, 'w') as f:
            f.write(f"342 {len(fem_data.elements.ids)}\n")
        pd.DataFrame(np.concatenate([
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN1'),
                is_engineering=True),
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN2'),
                is_engineering=True),
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN3'),
                is_engineering=True),
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN4'),
                is_engineering=True),
        ], axis=1), index=fem_data.elements.ids).to_csv(
            istrain_file_name, sep=' ', index=True, header=False,
            mode='a')

        if RUN_FISTR:
            os.system(f"cd {write_dir_name} "
                      + "&& fistr1 > /dev/null 2>&1")
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodes.data
                + written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                original_nodes)

    def test_invert_strain(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/invert_strain',
            read_npy=False, save=False)

        write_dir_name = Path('tests/data/fistr/write_invert_strain')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)

        fem_data.elemental_data.pop(
            'linear_thermal_expansion_coefficient', None)
        fem_data.elemental_data.pop(
            'ORIENTATION', None)
        fem_data.elemental_data.pop(
            'INITIAL_TEMPERATURE', None)
        fem_data.elemental_data.pop(
            'CNT_TEMPERATURE', None)
        original_nodes = fem_data.nodes.data
        fem_data.nodes.data = fem_data.nodes.data \
            + fem_data.nodal_data.get_attribute_data('DISPLACEMENT')
        fem_data.settings['solution_type'] = 'EPS2DISP'
        fem_data.write('fistr', write_dir_name / 'mesh')

        istrain_file_name = write_dir_name / 'istrain.dat'
        with open(istrain_file_name, 'w') as f:
            f.write(f"342 {len(fem_data.elements.ids)}\n")
        pd.DataFrame(np.concatenate([
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN1'),
                is_engineering=True),
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN2'),
                is_engineering=True),
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN3'),
                is_engineering=True),
            functions.invert_strain(
                fem_data.elemental_data.get_attribute_data('GaussSTRAIN4'),
                is_engineering=True),
        ], axis=1), index=fem_data.elements.ids).to_csv(
            istrain_file_name, sep=' ', index=True, header=False,
            mode='a')

        if RUN_FISTR:
            os.system(f"cd {write_dir_name} "
                      + "&& fistr1 > /dev/null 2>&1")
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodes.data
                + written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                original_nodes, decimal=3)

    def test_calculate_symmetric_matrices_from_eigens(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/lte_direction',
            read_npy=False, save=False)
        lte, directions, lte_vec = functions.calculate_principal_components(
            fem_data.elemental_data.get_attribute_data('lte_full'),
            from_engineering=True)

        calculated_lte_full = functions.calculate_array_from_eigens(
            lte, directions, to_engineering=True)
        np.testing.assert_almost_equal(
            calculated_lte_full,
            fem_data.elemental_data.get_attribute_data('lte_full'))

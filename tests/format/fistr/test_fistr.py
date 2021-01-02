import unittest

import numpy as np

from femio.fem_data import FEMData


INP_FILE = 'tests/data/ucd/hex.inp'
FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_CNT_FILE = 'tests/data/fistr/thermal/hex.cnt'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'

NaN = np.nan


class TestFEMDataFISTR(unittest.TestCase):

    def test_fistr_node_ids(self):
        """Node IDs should be correctly stored."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        desired_node_ids = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        np.testing.assert_equal(
            fem_data.nodes.ids, desired_node_ids)

    def test_fistr_node_positions(self):
        """Node positions should be correctly stored."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        desired_node_positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0]])
        np.testing.assert_almost_equal(
            fem_data.nodes.data, desired_node_positions)

    def test_fistr_initial_temperature(self):
        """Initial temperature should be correctly stored."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        desired_temperature = np.atleast_2d([
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
            90.0, 100.0, 110.0, 120.0]).T
        np.testing.assert_almost_equal(
            fem_data.nodal_data['INITIAL_TEMPERATURE'].data,
            desired_temperature)

    def test_fistr_element_ids(self):
        """Element IDs should be correctly stored."""
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        desired_element_ids = np.array([1, 2])
        np.testing.assert_equal(
            fem_data.elements.ids, desired_element_ids)

    def test_fistr_element_components(self):
        """Element compoments should be correctly stored."""
        desired_element_components = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9, 10, 11, 12]
        ])
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        np.testing.assert_equal(
            fem_data.elements.data, desired_element_components)

    def test_fistr_material(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        np.testing.assert_almost_equal(
            fem_data.materials['Young_modulus'].loc['M1'].values,
            210000.0)
        np.testing.assert_almost_equal(
            fem_data.materials['Poisson_ratio'].loc['M1'].values, 0.35)
        np.testing.assert_almost_equal(
            fem_data.materials['density'].loc['M1'].values, 1.074e-9)
        np.testing.assert_almost_equal(
            fem_data.materials[
                'linear_thermal_expansion_coefficient'].loc['M1'].values,
            [0.001, 0.002, 0.003])
        np.testing.assert_almost_equal(
            fem_data.materials['Young_modulus'].loc['M2'].values, 100.0)
        np.testing.assert_almost_equal(
            fem_data.materials['Poisson_ratio'].loc['M2'].values, 0.25)
        np.testing.assert_almost_equal(
            fem_data.materials['density'].loc['M2'].values, 5.0e-9)
        np.testing.assert_almost_equal(
            fem_data.materials[
                'linear_thermal_expansion_coefficient'].loc['M2'].values,
            [-0.001, 0.004, 0.019])

    def test_fistr_displacement_ids(self):
        """Displacement IDs should be correctly stored."""
        desired_displacement_ids = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        np.testing.assert_equal(
            fem_data.nodal_data['DISPLACEMENT'].ids,
            desired_displacement_ids)

    def test_fistr_displacement_data(self):
        """Displacement data should be correctly stored."""
        desired_displacement_data = np.array([
            [0.0000000e+00,   0.0000000e+00,  0.0000000e+00],
            [-4.4284366e-02,  9.9999999e-01,  9.9999999e-01],
            [-1.0798443e+00,  9.6276728e-01, -9.9999999e-01],
            [-9.7777954e-01, -3.8912951e-02, -1.9827414e+00],
            [-9.1155894e-01,  2.0895053e+00, -4.0560031e-02],
            [-1.0721559e+00,  3.0814522e+00,  9.3240206e-01],
            [-2.1086070e+00,  2.9811416e+00, -1.0714801e+00],
            [-1.8887089e+00,  1.9908598e+00, -2.0812541e+00],
            [-1.5918726e+00,  6.5127537e+00, -7.1289412e-01],
            [-1.9560577e+00,  7.1817391e+00,  6.2508296e-02],
            [-3.3266281e+00,  6.0257792e+00, -1.9355183e+00],
            [-2.8866052e+00,  5.3869730e+00, -3.1508141e+00],
        ])
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        np.testing.assert_almost_equal(
            fem_data.nodal_data['DISPLACEMENT'].data,
            desired_displacement_data, decimal=3)

    def test_fistr_elemental_mises_ids(self):
        """Elemental MISES IDs should be correctly stored."""
        desired_nodal_mises_ids = np.array([1, 2])
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        np.testing.assert_almost_equal(
            fem_data.elemental_data['ElementalMISES'].ids,
            desired_nodal_mises_ids)

    def test_fistr_elemental_mises_data(self):
        """Elemental MISES data should be correctly stored."""
        desired_nodal_mises_data = np.array([
            [1.2858939470519477E+01],
            [5.7457980475401499E+01]
        ])
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_CNT_FILE, FISTR_RES_FILE])
        np.testing.assert_almost_equal(
            fem_data.elemental_data['ElementalMISES'].data,
            desired_nodal_mises_data, decimal=3)

    def test_file_extension_error(self):
        """ValueError should be raised when wrong file is passed."""
        with self.assertRaises(ValueError):
            FEMData.read_files('fistr', [INP_FILE])

    def test_pad_temperature(self):
        """Initial temperature should be padded with zero when missing."""
        tet2_data = FEMData.read_files(
            'fistr', ['tests/data/fistr/tet2/tet2.msh'])
        desired_temperature = np.array(
            [[10.], [20.], [30.], [40.], [0.], [0.], [0.], [0.], [0.], [0.]])
        np.testing.assert_almost_equal(
            tet2_data.nodal_data['INITIAL_TEMPERATURE'].data,
            desired_temperature)
        desired_elemental_temperature = np.array(
            [[10., 20., 30., 40., 0., 0., 0., 0., 0., 0.]])
        np.testing.assert_almost_equal(
            tet2_data.convert_nodal2elemental(
                'INITIAL_TEMPERATURE', ravel=True),
            desired_elemental_temperature)

    def test_read_heat_temperature(self):
        heat_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat/hex.msh', 'tests/data/fistr/heat/hex.cnt',
            'tests/data/fistr/heat/hex.res.0.100'])
        desired_initial_temperature = np.array([
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [10.0],
            [10.0],
            [0.0]
        ])
        np.testing.assert_almost_equal(
            heat_data.nodal_data['INITIAL_TEMPERATURE'].data[:10],
            desired_initial_temperature)

    def test_read_heat_material(self):
        heat_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat/hex.msh', 'tests/data/fistr/heat/hex.cnt'])
        desired_density = np.array([[2.0, 0.]])
        desired_specific_heat = np.array([[2.0, 0.]])
        desired_thermal_conductivity = np.array([[8.0, 0.]])
        np.testing.assert_almost_equal(
            heat_data.materials['density'].loc['TEST'][0],
            desired_density)
        np.testing.assert_almost_equal(
            heat_data.materials['specific_heat'].loc['TEST'][0],
            desired_specific_heat)
        np.testing.assert_almost_equal(
            heat_data.materials['thermal_conductivity'].loc['TEST'][0],
            desired_thermal_conductivity)

    def test_read_heat_nl_material(self):
        heat_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_nl', read_npy=False, save=False)
        np.testing.assert_almost_equal(
            heat_data.materials['density'].loc['STEEL'][0],
            np.array([
                [1.0, 0.0],
                [2.0, 4.0],
                [5.0, 10.0],
            ]))
        np.testing.assert_almost_equal(
            heat_data.materials['specific_heat'].loc['STEEL'][0],
            np.array([
                [1.0, 0.0],
                [3.0, 4.0],
                [4.0, 10.0],
            ]))
        np.testing.assert_almost_equal(
            heat_data.materials['thermal_conductivity'].loc['STEEL'][0],
            np.array([
                [1.0, 0.0],
                [5.0, 4.0],
                [10.0, 10.0],
            ]))
        np.testing.assert_almost_equal(
            heat_data.materials['density'].loc['TEST'][0],
            np.array([
                [10.0, 0.0],
            ]))
        np.testing.assert_almost_equal(
            heat_data.materials['specific_heat'].loc['TEST'][0],
            np.array([
                [100.0, 0.0],
            ]))
        np.testing.assert_almost_equal(
            heat_data.materials['thermal_conductivity'].loc['TEST'][0],
            np.array([
                [1000.0, 0.0],
            ]))

    def test_read_heat_nl_tensor(self):
        heat_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_nl_tensor',
            read_npy=False, save=False)
        np.testing.assert_almost_equal(
            heat_data.materials['density'].loc['STEEL'][0],
            np.array([
                [1.0, 0.0],
                [2.0, 4.0],
                [5.0, 10.0],
            ]))
        np.testing.assert_almost_equal(
            heat_data.materials['specific_heat'].loc['STEEL'][0],
            np.array([
                [1.0, 0.0],
                [3.0, 4.0],
                [4.0, 10.0],
            ]))
        np.testing.assert_almost_equal(
            heat_data.materials['thermal_conductivity_full'].loc['STEEL'][0],
            np.array([
                [0.2, 0.2, 0.2, 0.1, 0.0, 0.0, -1.0],
                [0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 1.0]
            ]))

    def test_read_heat_nl_time_series(self):
        time_series_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_nl', read_npy=False, save=False,
            time_series=True)
        heat_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_nl', read_npy=False, save=False)
        np.testing.assert_almost_equal(
            time_series_data.nodal_data.get_attribute_data('TEMPERATURE')[-1],
            heat_data.nodal_data.get_attribute_data('TEMPERATURE'))
        np.testing.assert_array_equal(
            time_series_data.nodal_data.get_attribute_ids('TEMPERATURE'),
            heat_data.nodal_data.get_attribute_ids('NODE'))
        np.testing.assert_array_equal(
            time_series_data.settings['time_steps'],
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    def test_read_static_time_series(self):
        time_series_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/static_time_series',
            read_npy=False, save=False, time_series=True)
        last_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/static_time_series',
            read_npy=False, save=False)
        np.testing.assert_almost_equal(
            time_series_data.nodal_data.get_attribute_data('DISPLACEMENT')[-1],
            last_data.nodal_data.get_attribute_data('DISPLACEMENT'))
        np.testing.assert_array_equal(
            time_series_data.elemental_data.get_attribute_data(
                'ElementalSTRAIN')[-1],
            last_data.elemental_data.get_attribute_data('ElementalSTRAIN'))
        np.testing.assert_array_equal(
            time_series_data.settings['time_steps'], [0, 5, 10])

    def test_read_element_egrp(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/element_egrp/plate.msh'])
        np.testing.assert_array_equal(
            fem_data.element_groups['E1'], [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(
            fem_data.element_groups['E2'], [7, 8, 9])

    def test_eliminate_blanck(self):
        """Data should be correctly loaded even with blanck lines."""
        FEMData.read_files('fistr', [
            'tests/data/fistr/with_blank/hex.msh'])

    def test_read_heat_steady(self):
        """Steady heat computation should be correctly loaded."""
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat_steady/hex.msh',
            'tests/data/fistr/heat_steady/hex.cnt'])
        ids = fem_data.constraints['fixtemp'].ids
        data = fem_data.constraints['fixtemp'].data
        np.testing.assert_array_equal(ids, [
            16,
            32,
            48,
            64,
            114,
            462,
        ])
        np.testing.assert_almost_equal(data, [
            [10.],
            [10.],
            [10.],
            [10.],
            [1.],
            [0.],
        ])

    def test_fistr_cload(self):
        """Cload condition should be read correctly."""
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/cload/hex.msh',
            'tests/data/fistr/cload/hex.cnt'])
        ids = fem_data.constraints['cload'].ids
        data = fem_data.constraints['cload'].data
        np.testing.assert_array_equal(np.unique(ids), np.unique([
            1, 4, 7, 10, 13, 16, 19, 22, 25, 3, 6, 9, 12, 15, 18, 21, 24, 27,
        ]))
        np.testing.assert_almost_equal(data, [
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [1., np.nan, np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [np.nan, 1., np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [-1., np.nan, np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
            [np.nan, -1., np.nan],
        ])

    def test_fistr_included_boundary(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/structure_inc_bnd', read_npy=False,
            save=False)
        actual_boundary = fem_data.constraints['boundary'].data
        desired_boundary = np.array([
            [0., 0., 0.],
            [0., 0., np.nan],
            [np.nan, 0., 0.]])
        np.testing.assert_almost_equal(actual_boundary, desired_boundary)

    def test_fistr_lte_full(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_full', read_npy=False,
            save=False)
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('lte_full'),
            np.array([
                [1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3],
                [2.1e-3, 2.2e-3, 2.3e-3, 2.4e-3, 2.5e-3, 2.6e-3],
                [3.1e-3, 3.2e-3, 3.3e-3, 3.4e-3, 3.5e-3, 3.6e-3],
                [4.1e-3, 4.2e-3, 4.3e-3, 4.4e-3, 4.5e-3, 4.6e-3],
            ])
        )

    def test_fistr_quad(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/quad', read_npy=False,
            save=False)
        self.assertEqual(fem_data.elements.element_type, 'quad')
        np.testing.assert_array_equal(
            fem_data.elements.data,
            np.array([[1, 2, 3, 4], [4, 3, 5, 6]]))

    def test_fistr_tri(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tri', read_npy=False,
            save=False)
        self.assertEqual(fem_data.elements.element_type, 'tri')
        np.testing.assert_array_equal(
            fem_data.elements.data,
            np.array([[1, 2, 3], [2, 3, 4]]))

    def test_fistr_res_multilines(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_res_multilines',
            read_npy=False, save=False)
        desired_nodal_mises = np.array([[
            1.0898350105447997E+04,
            1.3781943598714706E+04,
            1.4294185117208786E+04,
            1.8647204885291405E+04,
            9.3792804821633508E+03,
            7.3153108344243292E+03,
            6.7785975728721169E+03,
            5.1787984147854413E+03,
            7.4483454706158540E+01,
            6.4159304887717994E+01,
            4.9329405740780402E+01,
            2.9528125358349683E+01,
        ]]).T
        np.testing.assert_almost_equal(
            fem_data.nodal_data['NodalMISES'].data, desired_nodal_mises)
        desired_elemental_mises = np.array([
            [1.2858939470519477E+01],
            [5.7457980475401499E+01],
        ])
        np.testing.assert_almost_equal(
            fem_data.elemental_data['ElementalMISES'].data,
            desired_elemental_mises)

    def test_read_topopt(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/topopt',
            read_npy=False, save=False)
        self.assertTrue('res.0.26', fem_data.file_names)
        desired_vf_head = np.array([[
            1.0000000000000000E+00,
            1.0000000000000000E+00,
            1.0000000000000000E+00,
            1.0000000000000000E+00,
            1.0000000000000000E+00,
            1.0000000000000000E+00,
            1.0000000000000001E-05,
            1.0000000000000001E-05,
            1.0000000000000001E-05,
        ]]).T
        np.testing.assert_almost_equal(
            fem_data.elemental_data['VF'].data[:9],
            desired_vf_head)

    def test_read_no_elemental_res(self):
        FEMData.read_directory(
            'fistr', 'tests/data/fistr/no_elemental_res',
            read_npy=False, save=False)

    def test_read_data_with_useless_nodes(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/useless_nodes',
            read_npy=False, save=False)
        np.testing.assert_array_equal(
            fem_data.nodes.ids,
            np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13])
        )

    def test_read_data_with_useless_nodes_2(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/useless_nodes_2',
            read_npy=False, save=False)
        np.testing.assert_array_equal(
            fem_data.nodes.ids,
            np.unique(fem_data.elements.data))

    def test_read_spring(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/spring',
            read_npy=False, save=False)

        desired_spring_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(
            fem_data.constraints['spring'].ids,
            desired_spring_ids)

        desired_spring_data = np.array([
            [1.1e-6, NaN, NaN],
            [NaN, 1.2e-6, NaN],
            [NaN, NaN, 1.3e-6],
            [2.1e-6, NaN, NaN],
            [NaN, 2.2e-6, NaN],
            [NaN, NaN, 2.3e-6],
            [3.1e-6, NaN, NaN],
            [NaN, 3.2e-6, NaN],
            [NaN, NaN, 3.3e-6],
        ])
        np.testing.assert_almost_equal(
            fem_data.constraints['spring'].data,
            desired_spring_data)

    def test_read_mixture_solid_elements(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid',
            read_npy=False, save=True)
        np.testing.assert_array_equal(
            fem_data.elements['hex'].data,
            np.array([
                [1, 2, 4, 3, 5, 6, 8, 7],
                [13, 3, 4, 14, 15, 7, 8, 16],
            ])
        )
        np.testing.assert_array_equal(
            fem_data.elements['tet'].data,
            np.array([
                [12, 2, 6, 9],
            ])
        )
        np.testing.assert_array_equal(
            fem_data.elements.ids,
            [1, 3, 4, 8])

        desired_elements = [
            np.array([1, 2, 4, 3, 5, 6, 8, 7]),
            np.array([12, 2, 6, 9]),
            np.array([1, 5, 17, 2, 6, 12]),
            np.array([13, 3, 4, 14, 15, 7, 8, 16]),
        ]
        for e, de in zip(fem_data.elements.data, desired_elements):
            np.testing.assert_array_equal(e.data, de)

        desired_modulus = np.array([[10., 30., 40., 80.]]).T

        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('Young_modulus'),
            desired_modulus)

    def test_fistr_read_spring_boundary(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/spring_boundary',
            read_npy=False, save=False)
        np.testing.assert_array_equal(
            fem_data.elements['hex'].data,
            np.array([
                [1, 2, 3, 4, 5, 6, 7, 8],
            ])
        )
        np.testing.assert_array_equal(
            fem_data.elements['spring'].data,
            np.array([
                [1, 11],
                [2, 12],
                [3, 13],
                [4, 14],
                [5, 15],
                [6, 16],
                [7, 17],
                [8, 18],
            ])
        )
        np.testing.assert_array_equal(
            fem_data.elements.ids,
            [1, 11, 12, 13, 14, 15, 16, 17, 18])

        desired_elements = [
            np.array([1, 2, 3, 4, 5, 6, 7, 8]),
            np.array([1, 11]),
            np.array([2, 12]),
            np.array([3, 13]),
            np.array([4, 14]),
            np.array([5, 15]),
            np.array([6, 16]),
            np.array([7, 17]),
            np.array([8, 18]),
        ]
        for e, de in zip(fem_data.elements.data, desired_elements):
            np.testing.assert_array_equal(e.data, de)

        desired_modulus = np.array([
            [200000] + [20000000] * 8
        ]).T

        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('Young_modulus'),
            desired_modulus)

    def test_fistr_read_new_res_format(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3_new_res',
            read_npy=False, save=False)
        old_fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3',
            read_npy=False, save=False)
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('displacement'),
            old_fem_data.nodal_data.get_attribute_data('displacement'))

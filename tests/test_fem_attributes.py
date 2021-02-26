
import unittest

import numpy as np

from femio.fem_attribute import FEMAttribute
from femio.fem_attributes import FEMAttributes
from femio.fem_elemental_attribute import FEMElementalAttribute
from femio.fem_data import FEMData


FISTR_MSH_FILE = 'tests/data/fistr/thermal/hex.msh'
FISTR_RES_FILE = 'tests/data/fistr/thermal/hex.res.0.1'
FISTR_INP_FILE = 'tests/data/fistr/thermal/fistr_hex.inp'


class TestFEMAttributes(unittest.TestCase):

    def test_get_attribute_data(self):
        fem_attributes = FEMAttributes([
            FEMAttribute(
                f"data_{i}",
                ids=np.arange(5)+1,
                data=np.random.rand(5, 3))
            for i in range(3)])
        np.testing.assert_almost_equal(
            fem_attributes.get_attribute_data('data_1'),
            fem_attributes.data['data_1'].data)

    def test_overwrite_alias_nodal_data(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        fem_data.nodal_data.update_data(
            [3], {'t_init': 1000.}, allow_overwrite=True)
        np.testing.assert_almost_equal(
            fem_data.nodal_data['t_init'].loc[3].values, 1000.)

    def test_overwrite_alias_elemental_data(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        fem_data.elemental_data.update_data(
            [1], {'elemental_strain': np.arange(6)})

        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('elemental_strain'),
            np.arange(6))
        np.testing.assert_almost_equal(
            fem_data.elemental_data['elemental_strain']['hex'].loc[1].data[0],
            np.arange(6))

    def test_add_material(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        material_dict = {
            'Young_modulus': 1000., 'Poisson_ratio': 0.3, 'density': 5e10}
        fem_data.materials.update_data(
            'M_TEST', material_dict)
        np.testing.assert_array_equal(
            fem_data.materials['Young_modulus'].ids, ['M1', 'M2', 'M_TEST'])
        for k in fem_data.materials.keys():
            np.testing.assert_almost_equal(
                fem_data.materials[k].loc['M_TEST'].values, material_dict[k])

    def test_renew_material(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        material_dict = {
            'Young_modulus': 1000., 'Poisson_ratio': 0.3, 'density': 5e10}
        fem_data.materials.update_data(
            'M2', material_dict, allow_overwrite=True)
        for k in fem_data.materials.keys():
            np.testing.assert_almost_equal(
                fem_data.materials[k].loc['M2'].values, material_dict[k])

    def test_update_material(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE])
        fem_data.materials.update_data(
            'M1', {'Young_modulus': 999.}, allow_overwrite=True)
        np.testing.assert_almost_equal(
            fem_data.materials['Young_modulus'].loc['M1'].values, 999.)

    def test_add_section(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        fem_data.sections.update_data('M_ADDITIONAL', {'EGRP': 'E_ADDITIONAL'})
        self.assertEqual(len(fem_data.sections['EGRP']), 3)
        np.testing.assert_array_equal(
            fem_data.sections['EGRP']['M_ADDITIONAL'], ['E_ADDITIONAL'])

    def test_renew_section(self):
        fem_data = FEMData.read_files(
            'fistr', [FISTR_MSH_FILE, FISTR_RES_FILE])
        fem_data.sections.update_data(
            'M1', {'EGRP': 'E2'}, allow_overwrite=True)
        self.assertEqual(len(fem_data.sections['EGRP']), 2)
        self.assertEqual(fem_data.sections['EGRP']['M1'], ['E2'])

    def test_get_attribute_aliases(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_3', read_npy=False, save=False)
        popped_stress = fem_data.elemental_data.pop('elemental_stress')
        new_data = np.random.rand(*popped_stress.data.shape)
        fem_data.elemental_data.update({
            'elemental_stress': FEMElementalAttribute(
                'elemental_stress', ids=fem_data.elements.ids, data=new_data)})
        # Try to get attribute first without alias then next alias
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('elemental_stress'),
            new_data)

    def test_overwrite_boundary(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2', read_npy=False)
        ids = np.array([1, 2, 3])
        data = np.array([
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
            [3.1, 3.2, 3.3],
        ])
        fem_data.constraints.overwrite('boundary', ids=ids, data=data)
        np.testing.assert_array_equal(
            fem_data.constraints['boundary'].ids, ids)
        np.testing.assert_almost_equal(
            fem_data.constraints['boundary'].data, data)

    def test_update_data_success_when_key_missing(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2', read_npy=False,
            read_mesh_only=True)
        fem_data.materials.update_data(
            'MAT_HEAT', {
                'density': 1., 'specific_heat': 2.,
                'thermal_conductivity': 3.})

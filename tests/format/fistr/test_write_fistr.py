from glob import glob
import os
from pathlib import Path
import subprocess
import shutil
import unittest

import numpy as np

from femio.fem_attribute import FEMAttribute
# from femio.fem_attributes import FEMAttributes
from femio.fem_data import FEMData


RUN_FISTR = True


class TestWriteFistr(unittest.TestCase):

    def test_write_fistr_static(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/thermal/hex.msh',
            'tests/data/fistr/thermal/hex.cnt',
            'tests/data/fistr/thermal/hex.res.0.1'])

        write_file_name = 'tests/data/fistr/write_static/mesh'
        if os.path.isfile(write_file_name + '.msh'):
            os.remove(write_file_name + '.msh')
        if os.path.isfile(write_file_name + '.cnt'):
            os.remove(write_file_name + '.cnt')
        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        written_fem_data = FEMData.read_files('fistr', [
            write_file_name + '.msh',
            write_file_name + '.cnt'])
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data['ORIENTATION'].data,
            fem_data.elemental_data['ORIENTATION'].data)

        np.testing.assert_array_equal(
            list(fem_data.element_groups.keys()),
            list(written_fem_data.element_groups.keys()))
        for v1, v2 in zip(
                list(fem_data.element_groups.values()),
                list(written_fem_data.element_groups.values())):
            np.testing.assert_array_equal(v1, v2)

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=os.path.dirname(write_file_name), shell=True)
            written_fem_data_with_res = FEMData.read_files('fistr', [
                write_file_name + '.msh',
                write_file_name + '.cnt',
                write_file_name + '.res.0.1'])
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_static_id_not_from_1(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/thermal_id_not_from_1/hex.msh',
            'tests/data/fistr/thermal_id_not_from_1/hex.cnt',
            'tests/data/fistr/thermal_id_not_from_1/hex.res.0.1'])

        write_file_name = 'tests/data/fistr/write_static_id_not_from_1/mesh'
        if os.path.isfile(write_file_name + '.msh'):
            os.remove(write_file_name + '.msh')
        if os.path.isfile(write_file_name + '.cnt'):
            os.remove(write_file_name + '.cnt')
        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        written_fem_data = FEMData.read_files('fistr', [
            write_file_name + '.msh',
            write_file_name + '.cnt'])
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data['ORIENTATION'].data,
            fem_data.elemental_data['ORIENTATION'].data)

        np.testing.assert_array_equal(
            list(fem_data.element_groups.keys()),
            list(written_fem_data.element_groups.keys()))
        for v1, v2 in zip(
                list(fem_data.element_groups.values()),
                list(written_fem_data.element_groups.values())):
            np.testing.assert_array_equal(v1, v2)

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=os.path.dirname(write_file_name), shell=True)
            written_fem_data_with_res = FEMData.read_files('fistr', [
                write_file_name + '.msh',
                write_file_name + '.cnt',
                write_file_name + '.res.0.1'])
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_heat(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat/hex.msh',
            'tests/data/fistr/heat/hex.cnt',
            'tests/data/fistr/heat/hex.res.0.100'])

        write_file_name = 'tests/data/fistr/write_heat/mesh'
        if os.path.isfile(write_file_name + '.msh'):
            os.remove(write_file_name + '.msh')
        if os.path.isfile(write_file_name + '.cnt'):
            os.remove(write_file_name + '.cnt')
        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        written_fem_data = FEMData.read_files('fistr', [
            write_file_name + '.msh',
            write_file_name + '.cnt'])
        np.testing.assert_almost_equal(
            written_fem_data.nodal_data['INITIAL_TEMPERATURE'].data,
            fem_data.nodal_data['INITIAL_TEMPERATURE'].data
        )

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=os.path.dirname(write_file_name), shell=True)
            written_fem_data_with_res = FEMData.read_files('fistr', [
                write_file_name + '.msh',
                write_file_name + '.cnt',
                write_file_name + '.res.0.100'])
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['TEMPERATURE'].data,
                fem_data.nodal_data['TEMPERATURE'].data)

    def test_write_fistr_overwrite_error(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat/hex.msh',
            'tests/data/fistr/heat/hex.cnt',
            'tests/data/fistr/heat/hex.res.0.100'])
        with self.assertRaises(ValueError):
            fem_data.write(
                'fistr', file_name='tests/data/fistr/heat/hex')

    def test_write_fistr_heat_steady(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat_steady/hex.msh',
            'tests/data/fistr/heat_steady/hex.cnt',
            'tests/data/fistr/heat_steady/hex.res.0.1'])

        write_file_name = 'tests/data/fistr/write_heat_steady/mesh'
        if os.path.isfile(write_file_name + '.msh'):
            os.remove(write_file_name + '.msh')
        if os.path.isfile(write_file_name + '.cnt'):
            os.remove(write_file_name + '.cnt')
        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        written_fem_data = FEMData.read_files('fistr', [
            write_file_name + '.msh',
            write_file_name + '.cnt'])
        np.testing.assert_almost_equal(
            written_fem_data.constraints['fixtemp'].data,
            fem_data.constraints['fixtemp'].data
        )

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=os.path.dirname(write_file_name), shell=True)
            written_fem_data_with_res = FEMData.read_files('fistr', [
                write_file_name + '.msh',
                write_file_name + '.cnt',
                write_file_name + '.res.0.1'])
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['TEMPERATURE'].data,
                fem_data.nodal_data['TEMPERATURE'].data)

    def test_write_fistr_cload(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/cload/hex.msh',
            'tests/data/fistr/cload/hex.cnt'])

        write_file_name = 'tests/data/fistr/write_cload/mesh'
        if os.path.isfile(write_file_name + '.msh'):
            os.remove(write_file_name + '.msh')
        if os.path.isfile(write_file_name + '.cnt'):
            os.remove(write_file_name + '.cnt')
        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        written_fem_data = FEMData.read_files('fistr', [
            write_file_name + '.msh',
            write_file_name + '.cnt'])
        np.testing.assert_almost_equal(
            written_fem_data.constraints['cload'].data,
            fem_data.constraints['cload'].data
        )

        # if RUN_FISTR:
        #     os.system(f"cd {os.path.dirname(write_file_name)} "
        #               + "&& fistr1 > /dev/null 2>&1")
        #     written_fem_data_with_res = FEMData.read_files('fistr', [
        #         write_file_name + '.msh',
        #         write_file_name + '.cnt',
        #         write_file_name + '.res.0.1'])
        #     np.testing.assert_almost_equal(
        #         written_fem_data_with_res.nodal_data['TEMPERATURE'].data,
        #         fem_data.nodal_data['TEMPERATURE'].data)

    def test_write_fistr_static_overwrite(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/thermal/hex.msh',
            'tests/data/fistr/thermal/hex.cnt',
            'tests/data/fistr/thermal/hex.res.0.1'])

        write_file_name = 'tests/data/fistr/write_static_overwrite/mesh'
        if os.path.isfile(write_file_name + '.msh'):
            os.remove(write_file_name + '.msh')
        if os.path.isfile(write_file_name + '.cnt'):
            os.remove(write_file_name + '.cnt')

        data = np.random.rand(*fem_data.elemental_data.get_attribute_data(
            'lte').shape)
        fem_data.elemental_data.overwrite('lte', data)
        data = np.random.rand(*fem_data.elemental_data.get_attribute_data(
            'modulus').shape)
        fem_data.elemental_data.overwrite('modulus', data)
        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        written_fem_data = FEMData.read_files('fistr', [
            write_file_name + '.msh',
            write_file_name + '.cnt'])
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('lte'),
            fem_data.elemental_data.get_attribute_data('lte'))
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('modulus'),
            fem_data.elemental_data.get_attribute_data('modulus'))

    def test_write_boundary(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet2_bnd', read_npy=False, save=False)

        write_file_name = 'tests/data/fistr/write_tet2_bnd/mesh'
        write_dir_name = os.path.dirname(write_file_name)
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)

        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.constraints['boundary'].data,
            fem_data.constraints['boundary'].data)
        np.testing.assert_array_equal(
            written_fem_data.constraints['boundary'].ids,
            fem_data.constraints['boundary'].ids)

    def test_write_fistr_lte_full(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_full', read_npy=False,
            save=False)

        write_dir_name = 'tests/data/fistr/write_thermal_full'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write('fistr', write_dir_name + '/mesh')

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('lte_full'),
            fem_data.elemental_data.get_attribute_data('lte_full'),
        )

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_convert_lte_full_to_lte_local_1(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_full_easy', read_npy=False,
            save=False)

        write_dir_name = 'tests/data/fistr/write_thermal_convert_easy'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.convert_lte_global2local()
        np.testing.assert_almost_equal(
            fem_data.elemental_data.get_attribute_data('lte')[0] * 1e7,
            np.array([1.0, 2.0, 3.0])
        )
        fem_data.write('fistr', write_dir_name + '/mesh')

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.elemental_data.get_attribute_data(
                    'ElementalSTRAIN'),
                fem_data.elemental_data.get_attribute_data(
                    'ElementalSTRAIN'), decimal=5)

    def test_write_fistr_convert_lte_full_to_lte_local_2(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_full', read_npy=False,
            save=False)

        write_dir_name = 'tests/data/fistr/write_thermal_convert'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.convert_lte_global2local()
        fem_data.write('fistr', write_dir_name + '/mesh')

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.elemental_data.get_attribute_data(
                    'ElementalSTRAIN'),
                fem_data.elemental_data.get_attribute_data(
                    'ElementalSTRAIN'), decimal=5)

    def test_write_fistr_quad(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/quad', read_npy=False,
            save=False)

        write_dir_name = 'tests/data/fistr/write_quad'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write('fistr', write_dir_name + '/mesh')

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data.get_attribute_data(
                    'DISPLACEMENT'),
                fem_data.nodal_data.get_attribute_data(
                    'DISPLACEMENT'), decimal=5)

    def test_write_fistr_no_solition_type(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal', read_npy=False, save=False)
        fem_data.settings.pop('solution_type')

        write_dir_name = 'tests/data/fistr/write_static_wo_solution'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', write_dir_name + '/mesh', overwrite=True)

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data['ORIENTATION'].data,
            fem_data.elemental_data['ORIENTATION'].data)

        np.testing.assert_array_equal(
            list(fem_data.element_groups.keys()),
            list(written_fem_data.element_groups.keys()))
        for v1, v2 in zip(
                list(fem_data.element_groups.values()),
                list(written_fem_data.element_groups.values())):
            np.testing.assert_array_equal(v1, v2)

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_spring(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/spring', read_npy=False, save=False)

        write_dir_name = 'tests/data/fistr/write_spring'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', file_name=write_dir_name + '/mesh', overwrite=True)

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.constraints['spring'].data,
            fem_data.constraints['spring'].data
        )

    def test_write_fistr_mixed_shell(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_shell',
            read_npy=False, save=False)

        write_dir_name = 'tests/data/fistr/write_mixture_shell'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', write_dir_name + '/mesh', overwrite=True)

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_mixed_solid(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/mixture_solid',
            read_npy=False, save=False)

        write_dir_name = 'tests/data/fistr/write_mixture_solid'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', write_dir_name + '/mesh', overwrite=True)

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_spring_from_array(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_3', read_npy=False, save=False)
        fem_data.constraints = {
            'spring': FEMAttribute(
                'spring',
                fem_data.elements.data[0, :3], np.ones((3, 3)) * 1e-6)}

        write_dir_name = Path('tests/data/fistr/write_spring_from_array')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', file_name=write_dir_name / 'mesh')

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        original_constraints_data = fem_data.constraints['spring'].data
        written_constraints_data = written_fem_data.constraints['spring'].data
        np.testing.assert_almost_equal(
            original_constraints_data[~np.isnan(original_constraints_data)],
            written_constraints_data[~np.isnan(written_constraints_data)])

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=write_dir_name, shell=True)
            vis_files = glob(str(write_dir_name / '*.inp'))
            self.assertTrue(len(vis_files) > 0)

    def test_write_fistr_from_npy(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/npy/mixture_solid',
            read_npy=True, save=False)

        write_dir_name = 'tests/data/fistr/write_from_npy'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', write_dir_name + '/mesh', overwrite=True)

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_thermal_wo_density(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_wo_density',
            read_npy=False, save=False)

        write_dir_name = 'tests/data/fistr/write_thermal_wo_density'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', file_name=os.path.join(write_dir_name, 'mesh'))

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('lte'),
            fem_data.elemental_data.get_attribute_data('lte'))
        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_thermal_wo_density_material_overwritten(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/thermal_wo_density',
            read_npy=False, save=False)

        write_dir_name = 'tests/data/fistr/write_thermal_wo_density_overwrite'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.elemental_data.overwrite(
            'Young_modulus',
            fem_data.elemental_data.get_attribute_data('Young_modulus') + .1)
        fem_data.write(
            'fistr', file_name=os.path.join(write_dir_name, 'mesh'))

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('lte'),
            fem_data.elemental_data.get_attribute_data('lte'))
        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    def test_write_fistr_heat_no_visual(self):
        fem_data = FEMData.read_files('fistr', [
            'tests/data/fistr/heat/hex.msh',
            'tests/data/fistr/heat/hex.cnt',
            'tests/data/fistr/heat/hex.res.0.100'])
        fem_data.settings['write_visual'] = False

        write_file_name = 'tests/data/fistr/write_heat_no_vis/mesh'
        if os.path.isfile(write_file_name + '.msh'):
            os.remove(write_file_name + '.msh')
        if os.path.isfile(write_file_name + '.cnt'):
            os.remove(write_file_name + '.cnt')
        fem_data.write(
            'fistr', file_name=write_file_name, overwrite=True)

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=os.path.dirname(write_file_name), shell=True)
            vis_files = glob(write_file_name + '*.inp')
            self.assertTrue(len(vis_files) == 0)

    def test_write_fistr_tet_tet2(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/tet_3', read_npy=False, save=False)
        fem_data.settings['tet_tet2'] = True

        write_dir_name = Path('tests/data/fistr/write_tet_tet2')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', file_name=write_dir_name / 'mesh')
        with open(write_dir_name / 'hecmw_ctrl.dat') as f:
            lines = f.readlines()
        self.assertTrue(
            ('!TET_TET2, ON\n' in lines) or ('!TET_TET2,ON\n' in lines))

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=write_dir_name, shell=True)
            vis_files = glob(str(write_dir_name / '*.inp'))
            self.assertTrue(len(vis_files) > 0)

    def test_write_fistr_overwrite_material(self):
        fem_data = FEMData.read_directory(
            'fistr',  'tests/data/fistr/thermal_to_overwrite',
            read_npy=False, save=False)
        mean_pos = fem_data.convert_nodal2elemental(
            fem_data.nodal_data.get_attribute_data('node'), calc_average=True)
        new_lte_full = np.einsum(
                'ij,i->ij',
                fem_data.elemental_data.get_attribute_data('lte_full'),
                mean_pos[:, 0] + mean_pos[:, 1])
        fem_data.elemental_data.overwrite('lte_full', new_lte_full)

        write_dir_name = Path('tests/data/fistr/write_overtewrite')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', file_name=write_dir_name / 'mesh')

        written_fem_data = FEMData.read_directory(
            'fistr',  write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data(
                'lte_full'), new_lte_full)

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=write_dir_name, shell=True)
            vis_files = glob(str(write_dir_name / '*.inp'))
            self.assertTrue(len(vis_files) == 2)

    def test_read_heat_nl_material(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_nl', read_npy=False, save=False)
        heat_res = FEMData.read_files(
            'ucd', 'tests/data/fistr/heat_nl/hex_vis_psf.0100.inp')

        write_dir_name = Path('tests/data/fistr/write_heat_nl')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', file_name=write_dir_name / 'mesh')

        written_fem_data = FEMData.read_directory(
            'fistr',  write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.materials['thermal_conductivity'].values[0, 0],
            fem_data.materials['thermal_conductivity'].values[0, 0])

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=write_dir_name, shell=True)
            written_res_data = FEMData.read_files(
                'ucd', write_dir_name / 'mesh_vis_psf.0100.inp')
            np.testing.assert_almost_equal(
                written_res_data.nodal_data.get_attribute_data('TEMPERATURE'),
                heat_res.nodal_data.get_attribute_data('TEMPERATURE')
            )

    def test_read_heat_nl_tensor_material(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_nl_tensor',
            read_npy=False, save=False)
        heat_res = FEMData.read_files(
            'ucd', 'tests/data/fistr/heat_nl_tensor/hex_vis_psf.0100.inp')

        write_dir_name = Path('tests/data/fistr/write_heat_nl_tensor')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', file_name=write_dir_name / 'mesh')

        written_fem_data = FEMData.read_directory(
            'fistr',  write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.materials['thermal_conductivity_full'].values[
                0, 0],
            fem_data.materials['thermal_conductivity_full'].values[0, 0])

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=write_dir_name, shell=True)
            written_res_data = FEMData.read_files(
                'ucd', write_dir_name / 'mesh_vis_psf.0100.inp')
            np.testing.assert_almost_equal(
                written_res_data.nodal_data.get_attribute_data('TEMPERATURE'),
                heat_res.nodal_data.get_attribute_data('TEMPERATURE')
            )

    def test_read_heat_nl_tensor_material_long_value(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_nl_tensor',
            read_npy=False, save=False)

        write_dir_name = Path('tests/data/fistr/write_heat_nl_tensor_long')
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        raw_conductivity = np.array([
            [1., 2., 3., .01, .02, .03, -1.],
            [2., 4., 6., .02, .04, .06, 1.],
        ]) * 1e-3 * np.random.rand()
        conductivity = np.array(
            [[raw_conductivity, 0]], dtype=object)[:, [0]]
        fem_data.materials.update_data(
            'STEEL', {'thermal_conductivity_full': conductivity},
            allow_overwrite=True)
        fem_data.write('fistr', file_name=write_dir_name / 'mesh')

        written_fem_data = FEMData.read_directory(
            'fistr',  write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.materials['thermal_conductivity_full'].values[
                0, 0], conductivity[0, 0])

        if RUN_FISTR:
            subprocess.check_call(
                "fistr1", cwd=write_dir_name, shell=True)
            self.assertTrue(
                (write_dir_name / 'mesh_vis_psf.0100.inp').exists())

    def test_write_fistr_heat_static(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/heat_static', read_npy=False,
            save=False)

        write_dir_name = Path('tests/data/fistr/write_heat_static')
        if write_dir_name.exists():
            shutil.rmtree(write_dir_name)
        fem_data.write('fistr', write_dir_name / 'mesh')

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.elemental_data.get_attribute_data('lte_full'),
            fem_data.elemental_data.get_attribute_data('lte_full'),
        )

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

    # TODO: Update for new spring truss element type
    def test_write_spring_boundary(self):
        fem_data = FEMData.read_directory(
            'fistr', 'tests/data/fistr/spring_boundary',
            read_npy=False, save=False)

        write_dir_name = 'tests/data/fistr/write_spring_boundary'
        if os.path.exists(write_dir_name):
            shutil.rmtree(write_dir_name)
        fem_data.write(
            'fistr', write_dir_name + '/mesh', overwrite=True)

        written_fem_data = FEMData.read_directory(
            'fistr', write_dir_name, read_npy=False, save=False)
        np.testing.assert_almost_equal(
            written_fem_data.nodes.data, fem_data.nodes.data)

        if RUN_FISTR:
            subprocess.check_call("fistr1", cwd=write_dir_name, shell=True)
            written_fem_data_with_res = FEMData.read_directory(
                'fistr', write_dir_name, read_npy=False, save=False)
            np.testing.assert_almost_equal(
                written_fem_data_with_res.nodal_data['DISPLACEMENT'].data,
                fem_data.nodal_data['DISPLACEMENT'].data, decimal=5)

from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ...util import string_parser as st
from ... import config


class FistrWriter():

    def __init__(self, fem_data):
        self.fem_data = fem_data
        if 'solution_type' not in self.fem_data.settings \
                or self.fem_data.settings['solution_type'] is None:
            self.fem_data.settings['solution_type'] = 'STATIC'
            print('No solution type detected. Automatically set to STATIC.')

        self.fem_data.material_overwritten \
            = self.fem_data.overwritten_material_exists()
        return

    def write(self, file_name=None, *, overwrite=False, write_msh_only=False):
        """Write FEM data in FrontISTR format.

        Args:
            file_name: File name of the output file. If not fed,
                input_filename.out.ext will be the output file name.
            overwrite: Bool, if True, allow averwrite files (Default: False.)
            write_msh_only: Bool, if True, omit writing cnt file
                (Default: False.)
        """
        self.write_file_name_base = file_name
        self.overwrite = overwrite
        self.write_dir_name = self.write_file_name_base.parent
        self.write_msh_file = Path(str(self.write_file_name_base) + '.msh')
        self.write_cnt_file = Path(str(self.write_file_name_base) + '.cnt')
        self.write_hecmw_ctrl_file = self.write_dir_name / 'hecmw_ctrl.dat'

        self.write_msh()
        if write_msh_only:
            return self.write_msh_file
        else:
            self.write_cnt()
            self.write_hecmw_ctrl()
            return [self.write_msh_file, self.write_cnt_file,
                    self.write_hecmw_ctrl_file]

    def write_msh(self):
        if not self.overwrite and self.write_msh_file.exists():
            raise ValueError(f"File {self.write_msh_file} already exists")

        # Write basic information
        print('Start msh header')
        print(dt.now())
        self.write_string(self.write_msh_file, '!HEADER\n', mode='w')
        self.write_string(self.write_msh_file, 'Data written by femio\n')

        # Write nodes
        print('Start nodes')
        print(dt.now())
        self.write_data(
            self.write_msh_file, '!NODE\n',
            self.fem_data.nodes.ids, self.fem_data.nodes.data)

        # Write elements
        print('Start element')
        print(dt.now())
        for element_type, elements in self.fem_data.elements.items():
            fistr_element_type = self.detect_fistr_element_type(element_type)
            self.write_data(
                self.write_msh_file, f"!ELEMENT,TYPE={fistr_element_type}\n",
                elements.ids, elements.data, str_format='%d')

        # TODO: Write node groups

        # Write element groups
        print('Start element group')
        print(dt.now())
        if self.fem_data.material_overwritten:
            self.write_formatted_strings(
                self.write_msh_file,
                ('!EGROUP, EGRP=E{}\n', '{}'),
                (self.fem_data.elements.ids, self.fem_data.elements.ids))
        else:
            if len(self.fem_data.element_groups) > 0:
                all_keys = np.array(list(self.fem_data.element_groups.keys()))
                egrp_indices = ~(all_keys == 'ALL')
                if np.sum(egrp_indices) > 0:
                    values = np.concatenate(
                        np.array(
                            list(self.fem_data.element_groups.values()),
                            dtype=object)[egrp_indices])
                    if len(values) == len(self.fem_data.elements.ids) \
                            == len(self.fem_data.element_groups) - 1:
                        self.write_formatted_strings(
                            self.write_msh_file,
                            (
                                '!EGROUP, EGRP={}\n',
                                '{}'),
                            (
                                all_keys[egrp_indices], values))
                    else:
                        for k, value in self.fem_data.element_groups.items():
                            if k == 'ALL':
                                continue
                            self.write_data(
                                self.write_msh_file, f"!EGROUP, EGRP={k}\n",
                                value)

        # Write sections
        if self.fem_data.material_overwritten:
            print('Start section')
            print(dt.now())
            # Use elemental data instead of materials
            for element_type, elements in self.fem_data.elements.items():
                if len(elements) == 0:
                    continue

                if element_type in config.LINE_ELEMENT_NAMES:
                    type_ = 'SOLID'
                    params = ''
                elif element_type in config.SHELL_ELEMENT_NAMES:
                    type_ = 'SHELL'
                    params = '\n1.0, 1'  # TODO: Control shell parameters
                elif element_type in config.SOLID_ELEMENT_NAMES:
                    type_ = 'SOLID'
                    params = ''
                else:
                    raise ValueError(
                        f"Unknown element type: {element_type}")

                str_data = [
                    f"!SECTION, TYPE={type_}, EGRP=E{i}, "
                    + f"MATERIAL=M{i}{params}"
                    for i in elements.ids]
                self.write_string(
                    self.write_msh_file,
                    '\n'.join(str_data) + '\n')

            # Write materials
            if ('Young_modulus' in self.fem_data.elemental_data
                    and 'Poisson_ratio' in self.fem_data.elemental_data) or (
                        'Young_modulus' in self.fem_data.materials
                        and 'Poisson_ratio' in self.fem_data.materials):
                print('Start material')
                print(dt.now())
                self.write_material(self.write_msh_file)

        else:
            print('Start section')
            print(dt.now())
            if hasattr(self.fem_data, 'sections') \
                    and len(self.fem_data.sections) > 0:
                dict_param = {
                    'SHELL': '\n1.0, 1',
                    'SOLID': '',
                }
                params = [
                    dict_param[t] for t in np.ravel(
                        self.fem_data.sections.get_attribute_data('TYPE'))]
                self.write_formatted_strings(
                    self.write_msh_file,
                    (
                        '!SECTION, TYPE={},',
                        'EGRP={},',
                        'MATERIAL={}',
                        '{}'
                    ), (
                        np.ravel(
                            self.fem_data.sections.get_attribute_data('TYPE')),
                        np.ravel(
                            self.fem_data.sections.get_attribute_data('EGRP')),
                        self.fem_data.sections['EGRP'].ids,
                        params),
                )

            # Write materials
            if self.fem_data.elemental_data.get_n_material() > 0 or len(
                    self.fem_data.materials) > 0:
                print('Start material')
                print(dt.now())
                self.write_material(self.write_msh_file)

        # Write initial temperatures
        if 'INITIAL_TEMPERATURE' in self.fem_data.nodal_data:
            print('Start initial temp')
            print(dt.now())
            self.write_data(
                self.write_msh_file,
                '!INITIAL CONDITION, TYPE=TEMPERATURE\n',
                self.fem_data.nodal_data['INITIAL_TEMPERATURE'].ids,
                self.fem_data.nodal_data['INITIAL_TEMPERATURE'].data)

        self.write_string(self.write_msh_file, '!END\n')

    def write_cnt(self):
        if not self.overwrite and self.write_cnt_file.exists():
            raise ValueError(f"File {self.write_cnt_file} already exists")

        print('Start cnt header')
        print(dt.now())

        if self.fem_data.settings['solution_type'] == ['HEAT', 'HEATSTATIC']:
            if 'beta' not in self.fem_data.settings:
                self.fem_data.settings['beta'] = 1.0  # By default implicit
            additional_solution_type \
                = f", BETA={self.fem_data.settings['beta']}\n"
        else:
            additional_solution_type = '\n'

        self.write_string(
            self.write_cnt_file,
            '!VERSION\n5\n'
            + f"!SOLUTION, TYPE={self.fem_data.settings['solution_type']}"
            + additional_solution_type,
            mode='w')

        if 'frequency' in self.fem_data.settings:
            frequency = self.fem_data.settings['frequency']
        else:
            frequency = 1

        # Write heat setting
        if self.fem_data.settings['solution_type'] in ['HEAT']:
            if 'heat' in self.fem_data.settings \
                    and len(self.fem_data.settings['heat']) > 0:
                heat_setting = st.StringSeries.read_array(
                    self.fem_data.settings['heat'])[0] + '\n'
                if abs(self.fem_data.settings['heat'][0][0] - 0.) < 1e-5:
                    frequency = 1  # Steady
            else:
                heat_setting = ''
                frequency = 1  # Steady

            self.write_string(
                self.write_cnt_file,
                '!HEAT\n'
                + heat_setting)

        # Write basic information
        only_solid = True
        for element_type in self.fem_data.elements.keys():
            if element_type not in config.SOLID_ELEMENT_NAMES:
                only_solid = False
        if only_solid:
            additional_out = (
                '!OUTPUT_RES\n'
                + 'ESTRAIN,ON\n'
                + 'ESTRESS,ON\n'
                + 'EMISES,ON\n'
                + 'ISTRAIN,ON\n'
                + 'ITEMP,ON\n')
            additional_vis = (
                '!OUTPUT_VIS\n'
                + 'ESTRAIN,ON\n'
                + 'ESTRESS,ON\n'
                + 'EMISES,ON\n'
                + 'TEMPERATURE,ON\n')
        else:
            additional_out = (
                '!OUTPUT_RES\n'
                + 'DISP,ON\n')
            additional_vis = ''

        if 'output_res' in self.fem_data.settings:
            additional_out = additional_out \
                + self.fem_data.settings['output_res']
        if 'output_vis' in self.fem_data.settings:
            additional_vis = additional_vis \
                + self.fem_data.settings['output_vis']

        if 'write_visual' not in self.fem_data.settings:
            self.fem_data.settings['write_visual'] = True
        if self.fem_data.settings['write_visual']:
            write_visual_setting = \
                f"!WRITE,VISUAL, FREQUENCY={frequency}\n"
        else:
            write_visual_setting = ''
        self.write_string(
            self.write_cnt_file,
            f"!WRITE,RESULT, FREQUENCY={frequency}\n"
            + write_visual_setting + additional_out + additional_vis)

        # Write boundaries
        if 'boundary' in self.fem_data.constraints:
            print('Start boundary')
            print(dt.now())
            not_nan_indices = [
                np.where(~np.isnan(boundary))[0]
                for boundary in self.fem_data.constraints['boundary'].data]
            start_end = [[
                np.min(not_nan_index) + 1,
                np.min(not_nan_index) + len(not_nan_index)]
                         for not_nan_index in not_nan_indices]
            # Take mean for the case like [1., 1., nan]
            data = [np.mean(d[~np.isnan(d)])
                    for d in self.fem_data.constraints['boundary'].data]
            self.write_data(
                self.write_cnt_file,
                '!BOUNDARY\n',
                self.fem_data.constraints['boundary'].ids,
                start_end, data, str_format=['%d', '%.5E'])

        # Write cprings
        if 'spring' in self.fem_data.constraints:
            print('Start spring')
            print(dt.now())
            isnan = np.isnan(self.fem_data.constraints['spring'].data)
            spring_ids = self.fem_data.constraints['spring'].ids[
                np.where(~isnan)[0]]
            spring_directions = np.where(~isnan)[1] + 1
            spring_values = self.fem_data.constraints['spring'].data[~isnan]
            self.write_data(
                self.write_cnt_file,
                '!SPRING\n', spring_ids, spring_directions, spring_values,
                str_format=['%d', '%5E'])

        # Write cloads
        if 'cload' in self.fem_data.constraints:
            print('Start cload')
            print(dt.now())
            isnan = np.isnan(self.fem_data.constraints['cload'].data)
            cload_directions = np.where(~isnan)[1] + 1
            cload_values = self.fem_data.constraints['cload'].data[~isnan]
            self.write_data(
                self.write_cnt_file,
                '!CLOAD\n',
                self.fem_data.constraints['cload'].ids,
                cload_directions, cload_values, str_format=['%d', '%5E'])

        # Write fixtemps
        if 'fixtemp' in self.fem_data.constraints:
            print('Start cload')
            print(dt.now())
            self.write_data(
                self.write_cnt_file,
                '!FIXTEMP\n',
                self.fem_data.constraints['fixtemp'].ids,
                self.fem_data.constraints['fixtemp'].data)

        # Write cnt temperature boundary condition
        if 'CNT_TEMPERATURE' in self.fem_data.nodal_data:
            print('Start cnt temp')
            print(dt.now())
            t = self.fem_data.nodal_data['CNT_TEMPERATURE'].data
            mean = np.mean(t)
            if np.all(np.abs(t - mean) < 1e-10):
                self.write_string(
                    self.write_cnt_file,
                    f"!TEMPERATURE\nALL, {mean}\n")
            else:
                self.write_data(
                    self.write_cnt_file,
                    '!TEMPERATURE\n',
                    self.fem_data.nodal_data['CNT_TEMPERATURE'].ids,
                    self.fem_data.nodal_data['CNT_TEMPERATURE'].data)

        # Write local orientations
        if 'ORIENTATION' in self.fem_data.elemental_data:
            print('Start orientation')
            print(dt.now())
            self.write_formatted_strings(
                self.write_cnt_file,
                (
                    '!ORIENTATION, DEFINITION=COORDINATES, NAME=ORIENT{}\n',
                    '{}'),
                (
                    self.fem_data.elemental_data['ORIENTATION'].ids,
                    self.fem_data.elemental_data['ORIENTATION'].data))
            self.write_cnt_sections(self.write_cnt_file)

        # Write cnt material (linear thermal expansion coefficient)
        self.write_cnt_material()

        # Write step setting
        if 'step' in self.fem_data.settings \
                and len(self.fem_data.settings['step']) > 0:
            self.write_string(
                self.write_cnt_file,
                '!STEP\n' + self.fem_data.settings['step'])

        # Write trailing basic information
        print('Start cnt setting')
        print(dt.now())
        self.write_string(
            self.write_cnt_file,
            '!SOLVER,METHOD=MUMPS,PRECOND=1,ITERLOG=YES,TIMELOG=YES\n'
            + '100000000, 1\n'
            + '1.0e-08, 1.0, 0.0\n'
            + '!VISUAL, method=PSR\n'
            + '!surface_num = 1\n'
            + '!surface 1\n'
            + '!output_type = COMPLETE_REORDER_AVS\n'
            + '!END\n')

        return

    def write_cnt_material(self):
        if 'linear_thermal_expansion_coefficient' \
                in self.fem_data.elemental_data \
                or 'linear_thermal_expansion_coefficient' \
                in self.fem_data.materials:
            lte_type = 'ORTHOTROPIC'
            attribute_name = 'linear_thermal_expansion_coefficient'
            self._write_expansion(lte_type, attribute_name)
        elif 'linear_thermal_expansion_coefficient_full' \
                in self.fem_data.elemental_data \
                or 'linear_thermal_expansion_coefficient_full' \
                in self.fem_data.materials:
            lte_type = 'FULL'
            attribute_name = 'linear_thermal_expansion_coefficient_full'
            self._write_expansion(lte_type, attribute_name)
        elif 'thermal_conductivity_full' in self.fem_data.materials:
            self._write_conductibity()
        else:
            return

    def _write_conductibity(self):
        print('Start cnt conductivity material')
        print(dt.now())
        attribute_name = 'thermal_conductivity_full'
        material_names = self.fem_data.materials.get_attribute_ids(
            attribute_name)
        material_data = self.fem_data.materials.get_attribute_data(
            attribute_name)

        self.write_formatted_strings(
            self.write_cnt_file, (
                '!MATERIAL, NAME={}\n',
                f"!CONDUCTIVITY, TYPE=FULL, DEPENDENCIES=1\n" + '{}'
            ), (material_names, material_data))

    def _write_expansion(self, lte_type, attribute_name):
        print('Start cnt LTEC material')
        print(dt.now())
        if self.fem_data.material_overwritten:
            material_names = self.add_prefix(
                'M', self.fem_data.elements.ids)
            material_data = self.fem_data.elemental_data.get_attribute_data(
                attribute_name)
        else:
            material_names = self.fem_data.materials.get_attribute_ids(
                attribute_name)
            material_data = self.fem_data.materials.get_attribute_data(
                attribute_name)

        if self.fem_data.settings['solution_type'] == 'HEATSTATIC':
            elastic_property = '!ELASTIC\n1.0, 0.0\n'
        else:
            elastic_property = ''
        self.write_formatted_strings(
            self.write_cnt_file, (
                '!MATERIAL, NAME={}\n' + elastic_property,
                f"!EXPANSION_COEFF, TYPE={lte_type}\n" + '{}'
            ), (material_names, material_data))
        return

    def write_hecmw_ctrl(self):
        if not self.overwrite and self.write_hecmw_ctrl_file.exists():
            raise ValueError(
                f"File {self.write_hecmw_ctrl_file} already exists")

        if 'tet_tet2' in self.fem_data.settings \
                and self.fem_data.settings['tet_tet2']:
            str_tet_tet2 = '!TET_TET2, ON\n'
        else:
            str_tet_tet2 = ''

        self.write_string(
            self.write_hecmw_ctrl_file,
            '!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE\n'
            + self.write_msh_file.name + '\n'
            + str_tet_tet2
            + '!CONTROL, NAME=fstrCNT\n'
            + self.write_cnt_file.name + '\n'
            + '!RESULT, NAME=fstrRES, IO=OUT\n'
            + self.write_file_name_base.name + '.res\n'
            + '!RESULT, NAME=vis_out, IO=OUT\n'
            + self.write_file_name_base.name + '_vis\n',
            mode='w')

    def detect_fistr_element_type(self, element_type):
        if element_type == 'tri':
            return '731'
        elif element_type == 'quad':
            return '741'
        elif element_type == 'line':
            return '301'
        elif element_type == 'line2':
            return '302'
        elif element_type == 'spring':
            return '311'
        elif element_type == 'tet':
            return '341'
        elif element_type == 'tet2':
            return '342'
        elif element_type == 'prism':
            return '351'
        elif element_type == 'prism2':
            return '352'
        elif element_type == 'hex':
            return '361'
        elif element_type == 'hex2':
            return '362'
        else:
            raise ValueError(
                f"Unknown element type: {element_type}")

    def write_string(self, file_name, string, *, mode='a'):
        with open(file_name, mode) as f:
            f.write(string)

    def write_data(self, file_name, header, data_ids, *args, str_format=None):
        if str_format is None:
            str_format = '%.12E'
        data_str = st.StringSeries.read_array(data_ids).connect(
            st.StringSeries.connect_all(args, str_format=str_format))
        with open(file_name, 'a') as f:
            f.write(header)
            f.write('\n'.join(data_str))
            f.write('\n')

    def generate_formatted_string(self, format_strings, list_data):
        """Generate formatted strings.

        Parameters
        ----------
        format_strings: List[str]
            Formatting strings e.g. 'DATA={}'.
        list_data: List[numpy.ndarray]
            List of data to be formatted.

        Returns:
        formatted_string: str
        """
        if not isinstance(format_strings, (list, tuple)):
            format_strings = [format_strings]
        if not isinstance(list_data, (list, tuple)):
            list_data = [list_data]

        if len(format_strings) != len(list_data):
            raise ValueError(
                f"Lengths of format_strings and list_data should be the same.")
        i = 0
        data_frame = pd.DataFrame(st.StringSeries.read_array(list_data[0]))
        for d in list_data[1:]:
            s = st.StringSeries.read_array(d)
            s.name = str(i)
            data_frame = data_frame.join(s)
            i += 1
        formatters = [lambda x, s=s: s.format(x) for s in format_strings]

        def float_format(x):
            return f"{x:.8E}"
        return data_frame.to_string(
            formatters=formatters, index=False, header=False, max_cols=65536,
            max_colwidth=65536, float_format=float_format
        ).replace(' ', '') + '\n'

    def write_formatted_strings(
            self, file_name, format_strings, list_data):
        """Write formatted strings.

        Parameters
        ----------
        file_name: str or pathlib.Path
            File name to write.
        format_strings: List[str]
            Formatting strings e.g. 'DATA={}'.
        list_data: List[numpy.ndarray]
            List of data to be formatted.

        Returns:
        formatted_string: str
        """
        data_str = self.generate_formatted_string(format_strings, list_data)
        with open(file_name, 'a') as f:
            f.write(data_str)
        return

    def add_prefix(self, prefix, data):
        return [f"{prefix}{d}" for d in data]

    def extract_list_data(
            self, fem_attributes, attribute_names, *, id_prefix=None):
        """Extract data of attributes with the specified attribute_names.

        Parameters
        ----------
        fem_attributes: FEMAttributes
            FEMAttributes object to extracted data from.
        attribute_names: List[str]
            List of attribute names to extact data.
        id_prefix: str, optional [None]
            If fed, add prefix to ids.

        Returns
        -------
        extracted_data: List[numpy.array]
        """
        ids = list(fem_attributes.values())[0].ids
        if id_prefix is not None:
            ids = self.add_prefix(id_prefix, ids)
        return [ids] + [
            fem_attributes.get_attribute_data(attribute_name)
            for attribute_name in attribute_names]

    def write_material(self, file_name):
        solution_type = self.fem_data.settings['solution_type']
        n_item = max(
            self.fem_data.elemental_data.get_n_material(),
            self.fem_data.materials.get_n_material())
        if solution_type in ['STATIC', 'EPS2DISP', 'MESHDOCTOR']:
            if n_item == 0:
                return
            elif n_item == 2:
                if self.fem_data.material_overwritten:
                    list_material_data = self.extract_list_data(
                        self.fem_data.elemental_data,
                        ('Young_modulus', 'Poisson_ratio'),
                        id_prefix='M')
                else:
                    list_material_data = self.extract_list_data(
                        self.fem_data.materials,
                        ('Young_modulus', 'Poisson_ratio'))
                data_str = self.generate_formatted_string((
                    '!MATERIAL, NAME={}, ITEM=1\n',
                    '!ITEM=1, SUBITEM=2\n{},',
                    '{}'), list_material_data)
            elif n_item > 2:
                if 'density' in self.fem_data.materials:
                    if self.fem_data.material_overwritten:
                        list_material_data = self.extract_list_data(
                            self.fem_data.elemental_data,
                            ('Young_modulus', 'Poisson_ratio', 'density'),
                            id_prefix='M')
                    else:
                        list_material_data = self.extract_list_data(
                            self.fem_data.materials,
                            ('Young_modulus', 'Poisson_ratio', 'density'))

                    data_str = self.generate_formatted_string((
                        '!MATERIAL, NAME={}, ITEM=2\n',
                        '!ITEM=1, SUBITEM=2\n{},',
                        '{}\n',
                        '!ITEM=2, SUBITEM=1\n{}'), list_material_data)
                else:
                    # No density but with LTE
                    if self.fem_data.material_overwritten:
                        list_material_data = self.extract_list_data(
                            self.fem_data.elemental_data,
                            ('Young_modulus', 'Poisson_ratio'),
                            id_prefix='M')
                    else:
                        list_material_data = self.extract_list_data(
                            self.fem_data.materials,
                            ('Young_modulus', 'Poisson_ratio'))
                    data_str = self.generate_formatted_string((
                        '!MATERIAL, NAME={}, ITEM=1\n',
                        '!ITEM=1, SUBITEM=2\n{},',
                        '{}'), list_material_data)
            else:
                raise ValueError(
                    'Insufficient material information: '
                    f"{self.fem_data.materials.keys()}, "
                    f"{self.fem_data.elemental_data.keys()}")

        elif solution_type in ['HEAT', 'HEATSTATIC']:
            if self.fem_data.material_overwritten:
                raise NotImplementedError
            list_material_data = self.extract_list_data(
                self.fem_data.materials,
                ('density', 'specific_heat', 'thermal_conductivity'))
            data_str = self.generate_formatted_string(
                (
                    '!MATERIAL, NAME={}, ITEM=3\n',
                    '!ITEM=1, SUBITEM=1\n{}\n',
                    '!ITEM=2, SUBITEM=1\n{}\n',
                    '!ITEM=3, SUBITEM=1\n{}'
                ), list_material_data)
        else:
            return

        with open(file_name, 'a') as f:
            f.write(data_str)
        return

    def write_cnt_sections(self, file_name):
        data_str = [
            f"!SECTION,SECNUM={i+1},ORIENTATION=ORIENT{sec_id}\n"
            for i, sec_id
            in enumerate(self.fem_data.elemental_data['ORIENTATION'].ids)]
        with open(file_name, 'a') as f:
            f.write(''.join(data_str))

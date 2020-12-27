from datetime import datetime as dt
import os
import re

import numpy as np

from ...fem_attribute import FEMAttribute
from ...fem_attributes import FEMAttributes
from ...fem_data import FEMData
from ...fem_elemental_attribute import FEMElementalAttribute
from ...util import string_parser as st


class FrontISTRData(FEMData):
    """FEMEntity of FrontISTR version."""

    MATERIAL_PROPERTY_NAMES = {
        'STATIC': [
            'Young_modulus',
            'Poisson_ratio',
            'density',
            'linear_thermal_expansion_coefficient',
            'linear_thermal_expansion_coefficient_full',
        ],
        'HEAT': [
            'density',
            'specific_heat',
            'thermal_conductivity',
            'thermal_conductivity_full',
        ],
    }
    MATERIAL_PROPERTY_NAMES['EPS2DISP'] = MATERIAL_PROPERTY_NAMES['STATIC']
    MATERIAL_PROPERTY_NAMES['MESHDOCTOR'] = MATERIAL_PROPERTY_NAMES['STATIC']
    MATERIAL_PROPERTY_NAMES['STRESS2DISP'] = MATERIAL_PROPERTY_NAMES['STATIC']
    MATERIAL_PROPERTY_NAMES['TOPOPT'] = MATERIAL_PROPERTY_NAMES['STATIC']
    MATERIAL_PROPERTY_NAMES['HEATSTATIC'] = MATERIAL_PROPERTY_NAMES['HEAT']

    DICT_FISTR_ELEMENTS = {
        '731': 'tri',
        '741': 'quad',
        '301': 'line',
        '302': 'line2',
        '311': 'spring',
        '341': 'tet',
        '342': 'tet2',
        '351': 'prism',
        '352': 'prism2',
        '361': 'hex',
        '362': 'hex2',
        '641': 'line',
    }

    @classmethod
    def read_files(cls, file_names, read_mesh_only=False, time_series=False):
        """Initialize FrontISTREntity object.

        Args:
            file_names: femio.util.StringSeries
                File names.
            read_mesh_only: bool, optional [False]
                If true, read mesh (nodes and elements) and ignore
                material data, results and so on.
            time_series: bool, optional [False]
                If True, parse files as time series data.
        """
        obj = cls()
        obj.time_series = time_series
        obj.file_names = file_names
        obj.read_mesh_only = read_mesh_only

        str_data = obj._read_str_data()
        str_data = obj._expand_include(str_data, 'msh')
        str_data = obj._expand_include(str_data, 'cnt')
        print('Parsing data')
        obj._read_cnt_solution_type(str_data['cnt'])
        obj._read_msh(str_data['msh'])
        if not obj.read_mesh_only:
            obj._read_cnt(str_data['cnt'])
            obj.remove_useless_nodes()
            if obj.time_series:
                res_file_names = obj.file_names.find_match(r'\.res')
                time_steps = [
                    int(res_file_name.split('.')[-1])
                    for res_file_name in res_file_names]
                obj.settings['time_steps'] = time_steps
                if len(res_file_names) > 0:
                    list_attributes = [
                        obj._read_res(r) for r in str_data['res']]
                    list_nodal_attributes = [a[0] for a in list_attributes]
                    list_elemental_attributes = [a[1] for a in list_attributes]
                    obj.nodal_data.update_time_series(list_nodal_attributes)
                    obj.elemental_data.update_time_series(
                        list_elemental_attributes)

            else:
                nodal_attributes, elemental_attributes = obj._read_res(
                    str_data['res'])
                obj.nodal_data.update(nodal_attributes)
                obj.elemental_data.update(elemental_attributes)

        obj._resolve_assignments()
        return obj

    def _read_str_data(self):
        """Read each file specified the extension.

        Returns:
            dict with keys = string of extention, values = StringSeries
            object.
        """
        str_data = {
            'msh': self._read_files(r'\.msh'),
            'cnt': self._read_files(r'\.cnt', mandatory=False),
            'res': self._read_files(
                r'\.res', mandatory=False, pattern_ignore=None)}
        return str_data

    def _expand_include(self, str_data, ext):
        """Expand inclusion of files. Contents of included files are added on
        the bottom.

        Args:
            str_data: dict with keys = string showing ext,
                value = StringSeries object.
            ext: String to be used for accessing value from the str_data.
        Returns:
            str_data after expansion of inclusion.
        """
        if str_data[ext] is None:
            return str_data
        base_name = os.path.dirname(
            self.file_names.find_match(r'\.' + ext, convert_values=True)[0])
        str_data[ext] = str_data[ext].expand_include(
                r'!INCLUDE\s*,\s*INPUT\s*=\s*([\w\/\.]+)', base_name)
        return str_data

    def _read_msh(self, string_series):
        print('header_data: mesh')
        print(dt.now())
        header_data = string_series.to_header_data('!')
        print('finish')
        print(dt.now())
        self._read_nodes(header_data)
        self._read_elements(header_data)
        if not self.read_mesh_only:
            print('start node group')
            print(dt.now())
            self._read_node_groups(header_data)
            print('start element group')
            print(dt.now())
            self._read_element_groups(header_data)
            print('start material')
            print(dt.now())
            self._read_materials(header_data)
            print('start sections')
            print(dt.now())
            self._read_sections(header_data)
            print('start initial cond')
            print(dt.now())
            self._read_initial_condisions(header_data)
            print('msh finish')
            print(dt.now())

    def _read_cnt_solution_type(self, string_series):
        if string_series is None:
            print('cnt file not found. Assuming SOLUTION TYPE = STATIC')
            self.settings['solution_type'] = 'STATIC'
            return

        solutions = string_series.find_match('!SOLUTION')
        if len(solutions) != 1:
            print('Solution definition not found. '
                  'Assuming SOLUTION TYPE = STATIC')
            self.settings['solution_type'] = 'STATIC'
            return

        self.settings['solution_type'] = solutions.extract_captures(
            r'TYPE=(\w+)', convert_values=True)[0]
        print(f"Solution type: {self.settings['solution_type']}")

    def _read_cnt(self, string_series):
        if string_series is None:
            return
        print('header_data: cnt')
        print(dt.now())
        header_data = string_series.to_header_data('!')
        print('finish')
        print('start options')
        print(dt.now())
        self._read_cnt_options(header_data)
        print('start cnt mat')
        print(dt.now())
        self._read_cnt_materials(header_data)
        print('start cnt section')
        print(dt.now())
        self._read_cnt_sections(header_data)
        print('start cnt local')
        print(dt.now())
        self._read_cnt_locals(header_data)
        print('start cnt spring')
        print(dt.now())
        self._read_cnt_springs(header_data)
        print('start cnt temp')
        print(dt.now())
        self._read_cnt_temperatures(header_data)
        print('start cnt bound')
        print(dt.now())
        self._read_cnt_boundaries(header_data)
        print('start cnt cload')
        print(dt.now())
        self._read_cnt_cload(header_data)
        print('start cnt fixtemp')
        print(dt.now())
        self._read_cnt_fixtemps(header_data)
        print('cnt finish')
        print(dt.now())

    def _read_res(self, string_series):
        if string_series is None:
            return {}, {}
        print('start res split')
        print(dt.now())
        nodal_data, elemental_data = self._split_series(string_series)
        print('start nodal res')
        print(dt.now())
        nodal_attributes = self._parse_res(nodal_data, len(self.nodes.ids))
        print('start elemental res')
        print(dt.now())
        elemental_attributes = self._parse_res(
            elemental_data, len(self.elements.ids), is_elemental=True)
        print('res finish')
        print(dt.now())
        return nodal_attributes, elemental_attributes

    def _resolve_assignments(self):
        print('start resolve mat')
        print(dt.now())
        self._resolve_assignments_materials()
        print('finish resolve assignments')
        print(dt.now())
        return

    def _resolve_assignments_materials(self):
        ids = self._extract_ids_from_sections()
        if not np.any(ids):
            return
        property_names = self.materials.keys()
        for property_name in property_names:
            materials = self._extract_material_values(property_name)
            self.elemental_data[property_name] = \
                self.elements.generate_elemental_attribute(
                    property_name, ids, materials)
        return

    def _extract_ids_from_sections(self):
        if len(self.sections) == 0:
            list_ids = [self.elements.ids]
        else:
            list_ids = [
                self.element_groups[section_value]
                for section_value
                in self.sections.get_attribute_data('EGRP')[:, 0]]
        return np.concatenate(list_ids)

    def _extract_material_values(self, property_name):
        material = self.materials[property_name]
        return np.concatenate([
            np.repeat(
                np.atleast_2d(material.loc[material_name].values),
                len(self.element_groups[element_group_name]), axis=0)
            for material_name, element_group_name
            in zip(self.sections['EGRP'].ids, self.sections['EGRP'].data[:, 0])
        ])

    def _extract_setting_value_unit(self, item_name):
        set_unit = set([v.unit
                        for v in self.settings[item_name].values()])
        if len(set_unit) == 1:
            return set_unit.pop()
        else:
            raise ValueError(
                'Unit of each propery looks different: {}'.format(
                    item_name))

    def _read_nodes(self, header_data):
        series = header_data.extract_data('!NODE')
        self.nodes = \
            series.to_fem_attribute(
                name='NODE', id_column=0, slice_data_columns=slice(1, 4))

    def _read_elements(self, header_data):
        element_types = header_data.extract_headers(
            '!ELEMENT').extract_captures(r'TYPE=(\w+)')
        if np.all(element_types == element_types.iloc[0]):
            # Uniform element
            et = element_types.iloc[0]
            element_type = self._convert_fistr_element_type(et)
            series = header_data.extract_data('!ELEMENT')
            self.elements = FEMElementalAttribute(
                'ELEMENT', {
                    element_type:
                    series.to_fem_attribute(
                        name='ELEMENT', id_column=0,
                        slice_data_columns=slice(1, None),
                        data_type=int)
                })
            return
        else:
            # Mixed element
            element_contents = header_data.extract_data(
                '!ELEMENT', concatenate=False)
            dict_id = {element_type: [] for element_type in element_types}
            dict_data = {element_type: [] for element_type in element_types}
            # Deal with case in which the same type appears multiple times in
            # the same file.
            for element_type, element_content in zip(
                    element_types, element_contents):
                int_content = element_content.to_values(data_type=int)
                dict_id[element_type].append(int_content[:, 0])
                dict_data[element_type].append(int_content[:, 1:])

            self.elements = FEMElementalAttribute(
                'ELEMENT', {
                    self._convert_fistr_element_type(element_type):
                    FEMAttribute(
                        name='ELEMENT',
                        ids=np.concatenate(dict_id[element_type]),
                        data=np.concatenate(dict_data[element_type]))
                    for element_type in element_types})
            return

    def _convert_fistr_element_type(self, fistr_element_type):
        if fistr_element_type in self.DICT_FISTR_ELEMENTS:
            return self.DICT_FISTR_ELEMENTS[fistr_element_type]
        else:
            raise ValueError(
                f"Unknown FrontISTR element type: {fistr_element_type}")

    def _read_node_groups(self, header_data):
        # Make 'ALL' node group
        self.node_groups.update({'ALL': self.nodes.ids})

        ngrps = header_data.extract_headers(
            '!NGROUP').extract_captures(r'NGRP=(\w+)')
        series = header_data.extract_data(
            '!NGROUP', concatenate=False)
        self.node_groups.update(
            {n: l.to_values(data_type=int, to_rank1=True) for n, l
             in zip(ngrps, series)})

    def _read_element_groups(self, header_data):
        # Make 'ALL' element group
        self.element_groups.update({'ALL': self.elements.ids})
        elements = st.HeaderData.extract_headers(header_data, '!ELEMENT')
        egrps = elements.extract_captures(r'EGRP=(\w+)')
        if len(egrps) > 0:
            # Case EGRP is defined in the !ELEMENT headers
            if len(egrps) == len(self.elements.ids):
                # NOTE: Assume each element corresponding to each element group
                #       with the same order
                self.element_groups.update(
                    {e: np.array([i])
                     for e, i in zip(egrps, self.elements.ids)})
            else:
                series = header_data.extract_data(
                    '!ELEMENT', concatenate=False)
                self.element_groups.update({
                    e: l.split_vertical(1)[0].to_values(
                        data_type=int, to_rank1=True)
                    for e, l in zip(egrps, series)})
        else:
            # Case EGRP is defined in the !EGROUP headers
            egrps = st.HeaderData.extract_headers(
                header_data, '!EGROUP').extract_captures(r'EGRP=(\w+)')
            series = header_data.extract_data(
                '!EGROUP', concatenate=False)
            self.element_groups.update(
                {e: l.to_values(data_type=int, to_rank1=True) for e, l
                 in zip(egrps, series)})
        if len(egrps) == 0:
            return

    def _read_materials(self, header_data):
        mats = st.HeaderData.extract_headers(header_data, '!MATERIAL')
        if len(mats) == 0:
            return
        # NOTE: Assume all materials have the same number of items
        items = mats.iloc[0:1].extract_captures(r'ITEM=(\d+)').to_values(
            data_type=int, to_rank1=True)[0]
        names = mats.extract_captures(r'NAME=(\w+)')
        if len(names) == len(self.elements.ids):
            # NOTE: Assume each material corresponding to each element with the
            #       same order
            material_data = np.concatenate([
                header_data.extract_data(
                    fr"!ITEM\s*=\s*(?:{i+1})", concatenate=True).to_values()
                for i in range(items)], axis=1)
            property_names = self.MATERIAL_PROPERTY_NAMES[
                self.settings['solution_type']][:material_data.shape[1]]
            self.materials = FEMAttributes(
                names=property_names, ids=names, list_arrays=material_data.T)
        else:
            item_contents = [
                header_data.extract_data(
                    fr"!ITEM\s*=\s*(?:{i+1})", concatenate=False)
                for i in range(items)]
            item_contents_by_material = [
                item_content for item_content in zip(*item_contents)]
            self.materials.update(self._parse_materials(
                names, item_contents_by_material))

        return

    def _read_sections(self, header_data):
        sects = st.HeaderData.extract_headers(header_data, '!SECTION')
        if len(sects) == 0:
            return
        types = sects.extract_captures(r'TYPE=(\w+)')
        egrps = sects.extract_captures(r'EGRP=(\w+)')
        mats = sects.extract_captures(r'MATERIAL=(\w+)')
        self.sections = FEMAttributes(
            names=['TYPE', 'EGRP'], ids=mats, list_arrays=[types, egrps])
        return

    def _read_initial_condisions(self, header_data):
        inits = st.HeaderData.extract_headers(
            header_data, '!INITIAL CONDITION')
        if len(inits) == 0:
            return
        init_types = inits.extract_captures(r'TYPE=(\w+)')
        values = header_data.extract_data(
            '!INITIAL CONDITION', concatenate=False)
        for init_type, value in zip(init_types, values):
            extended_value = self._extend_assignments(value)
            self.nodal_data.update(
                {'INITIAL_' + init_type: extended_value.to_fem_attribute(
                    name='INITIAL_' + init_type, id_column=0,
                    slice_data_columns=slice(1, None))})
        # Pad missing initial temperature value with zero
        if 'INITIAL_TEMPERATURE' in self.nodal_data.keys():
            n_node = len(self.nodes.data)
            n_filled_temp = len(self.nodal_data['INITIAL_TEMPERATURE'].data)
            if (n_node != n_filled_temp):
                initial_temperatures = np.concatenate(
                        [self.nodal_data['INITIAL_TEMPERATURE'].data,
                         np.zeros([n_node - n_filled_temp, 1])])
                self.nodal_data['INITIAL_TEMPERATURE'] = FEMAttribute(
                    'INITIAL_TEMPERATURE', self.nodes.ids,
                    initial_temperatures)
                print('Enpty initial temperature found. Padded with zero.')

    def _read_cnt_options(self, header_data):
        heats = st.HeaderData.extract_headers(header_data, '!HEAT')
        if len(heats) > 0:
            heat_data = header_data.extract_data('!HEAT')
            self.settings.update({'heat': heat_data.to_values()})

        steps = st.HeaderData.extract_headers(header_data, '!STEP')
        if len(steps) > 0:
            step_data = header_data.extract_data('!STEP')
            str_step_data = '\n'.join(step_data.values) + '\n'
            self.settings.update({'step': str_step_data})
        return

    def _read_cnt_materials(self, header_data):
        mats = st.HeaderData.extract_headers(header_data, '!MATERIAL')
        if len(mats) == 0:
            return
        names = mats.extract_captures(r'NAME=(\w+)')

        exps = st.HeaderData.extract_headers(header_data, '!EXPANSION_COEFF')
        if len(exps) > 0:
            self._read_cnt_expansions(header_data, names, exps)

        conductivities = st.HeaderData.extract_headers(
            header_data, '!CONDUCTIVITY')
        if len(conductivities) > 0:
            self._read_cnt_conductivities(header_data, names, conductivities)

        return

    def _read_cnt_expansions(self, header_data, names, exps):
        exp_types = exps.extract_captures(r'TYPE=(\w+)')
        exp_type = exp_types.iloc[0]
        if not np.all(exp_types == exp_type):
            raise NotImplementedError('Mixed type: FULL and  ORTHOTROPIC')

        values = header_data.extract_data('!EXPANSION_COEFF').to_values()
        if exp_type == 'ORTHOTROPIC':
            property_name = 'linear_thermal_expansion_coefficient'
        elif exp_type == 'FULL':
            property_name = 'linear_thermal_expansion_coefficient_full'
        else:
            raise ValueError(f"Unknown type for !EXPANSION_COEFF: {exp_types}")
        self.materials.update({
            property_name: FEMAttribute(property_name, names, values)})
        return

    def _read_cnt_conductivities(self, header_data, names, conductivities):
        conductivity_types = conductivities.extract_captures(r'TYPE=(\w+)')
        if len(conductivities) > 1:
            raise NotImplementedError(
                f"# of materials of CONDUCTIVITY should be 1")
        conductivity_type = conductivity_types.iloc[0]
        if not np.all(conductivity_types == conductivity_type):
            raise NotImplementedError('Mixed type for CONDUCTIVITY in cnt')

        values = np.array(
            [[header_data.extract_data('!CONDUCTIVITY').to_values(), 0]],
            dtype=object)[:, 0, None]

        if conductivity_type == 'FULL':
            property_name = 'thermal_conductivity_full'
            # self.materials.pop('thermal_conductivity')
        else:
            raise ValueError(
                f"Unknown type for !CONDUCTIVITY: {conductivity_type}")
        self.materials.update({
            property_name: FEMAttribute(property_name, names, values)})
        return

    def _read_cnt_sections(self, header_data):
        sects = st.HeaderData.extract_headers(header_data, '!SECTION')
        if len(sects) == 0:
            return
        orients = sects.extract_captures(r'ORIENTATION=(\w+)')
        # Create map from orients to material name to make searching eqsier
        self.sections['ORIENTATION'] = FEMAttribute(
            'ORIENTATION', ids=orients, data=self.sections['EGRP'].ids)
        return

    def _read_cnt_locals(self, header_data):
        orients = st.HeaderData.extract_headers(
            header_data, '!ORIENTATION')
        if len(orients) == 0:
            return
        names = orients.extract_captures(r'NAME=(\w+)')
        material_names = np.ravel(
            self.sections['ORIENTATION'].loc[names].values)
        list_values = header_data.extract_data(
            '!ORIENTATION').to_values()
        self.materials.update({
            'ORIENTATION':
            FEMAttribute('ORIENTATION', material_names, list_values)})
        return

    def _read_cnt_temperatures(self, header_data):
        temps = header_data.extract_data('!TEMPERATURE')
        if len(temps) == 0:
            return
        temp_data = self._extend_assignments(temps)
        self.nodal_data.update(
            {'CNT_TEMPERATURE':
             temp_data.to_fem_attribute('CNT_TEMPERATURE', 0, 1)})

    def _read_cnt_boundaries(self, header_data):
        bnds = header_data.extract_data('!BOUNDARY')
        if len(bnds) == 0:
            return
        bnd_data = self._extend_assignments(bnds)
        _ids, _starts, _ends, _values = bnd_data.split_vertical_all()
        ids = _ids.astype(int)
        starts = _starts.astype(int)
        ends = _ends.astype(int)
        values = _values.astype(float)
        data = np.empty((len(ids), 3))
        data[:] = np.nan
        for d, start, end, value in zip(data, starts, ends, values):
            d[start-1:end] = value
        self.constraints.update(
            {'boundary': FEMAttribute('boundary', ids, data)})

    def _read_cnt_springs(self, header_data):
        bnds = header_data.extract_data('!SPRING')
        if len(bnds) == 0:
            return
        bnd_data = self._extend_assignments(bnds)
        _ids, _directions, _values = bnd_data.split_vertical_all()
        ids = _ids.astype(int)
        directions = _directions.astype(int)
        values = _values.astype(float)
        data = np.empty((len(ids), 3))
        data[:] = np.nan
        for d, direction, value in zip(data, directions, values):
            d[direction - 1] = value
        self.constraints.update(
            {'spring': FEMAttribute('spring', ids, data)})

    def _read_cnt_cload(self, header_data):
        cloads = header_data.extract_data('!CLOAD')
        if len(cloads) == 0:
            return
        cload_data = self._extend_assignments(cloads)
        _ids, _dims, _values = cload_data.split_vertical_all()
        ids = _ids.astype(int)
        dims = _dims.astype(int)
        values = _values.astype(float)

        data = np.empty((len(ids), 3))
        data[:] = np.nan
        for d, dim, value in zip(data, dims, values):
            d[dim-1] = value
        self.constraints.update(
            {'cload': FEMAttribute('cload', ids, data)})

    def _read_cnt_fixtemps(self, header_data):
        fixtemps = header_data.extract_data('!FIXTEMP')
        if len(fixtemps) == 0:
            return
        fixtemp_data = self._extend_assignments(fixtemps)
        _ids, _temperatures = fixtemp_data.split_vertical_all()
        ids = _ids.astype(int)
        temperatures = _temperatures.astype(float)
        self.constraints.update(
            {'fixtemp': FEMAttribute('fixtemp', ids, temperatures)})

    def _extend_assignments(self, series):
        node_groups, values = series.find_match(
            r'^\s*[A-Za-z]').split_vertical(1)
        if len(node_groups) == 0:
            return series

        sl_ids = [st.StringSeries.read_array(self.node_groups[node_group])
                  for node_group in node_groups.strip()]
        sl_values = [
            st.StringSeries.read_array(np.array(
                [value for _ in range(len(sl_id))]))
            for value, sl_id in zip(values, sl_ids)]
        concatenated_data = st.StringSeries.concat(
            [i.connect(v) for i, v in zip(sl_ids, sl_values)]
            + [series.find_match(r'^\s*\d')])
        return concatenated_data

    def _parse_materials(self, names, item_contents):
        if self.settings['solution_type'] in [
                'STATIC', 'EPS2DISP', 'STRESS2DISP', 'TOPOPT']:
            properties = [
                st.StringSeries.read_array(np.ravel(np.concatenate(
                    [s.delimit() for s in item],
                    axis=1)))
                for item in item_contents]
            list_data_values = np.array([
                p.to_values(to_rank1=True) for p in properties]).T
        elif self.settings['solution_type'] in ['HEAT', 'HEATSTATIC']:
            # To force making object array, we make (s.to_values(), 0)
            list_data_values = np.swapaxes(np.array([
                [(s.to_values(), 0) for s in item]
                for item in item_contents], dtype=object), 0, 1)
            list_data_values = list_data_values[:, :, 0]
        else:
            raise ValueError(
                'Unsupported SOLUTION TYPE = '
                + f"{self.settings['solution_type']}")

        material_property_names = self.MATERIAL_PROPERTY_NAMES[
            self.settings['solution_type']][:len(list_data_values)]
        return FEMAttributes(
            names=material_property_names, ids=names,
            list_arrays=list_data_values)

    def _split_series(self, string_series):
        if len(string_series.find_match('TOTALTIME')) == 0:
            # Old res format
            content_start = 3
        else:
            content_start = 11
        res_contents = st.StringSeries.read_array(
            string_series[content_start:])
        ind_clusters = res_contents.indices_match_clusters(r'^[\*a-zA-Z]')

        if len(ind_clusters) > 1:
            # Both nodal and elemental data exist
            nodal_value_found = False
            i_go_back = 1
            while not nodal_value_found:
                if re.search(
                        r'E\+?-?\d+', res_contents[
                            ind_clusters[1][0] - i_go_back]):
                    break
                i_go_back += 1
            elemental_data_start = ind_clusters[1][0] - i_go_back + 1
            elemental_data = st.StringSeries.read_array(
                    res_contents[elemental_data_start:])
            nodal_data = st.StringSeries.read_array(
                res_contents[:elemental_data_start])

        else:
            # Only nodal data exists
            elemental_data = None
            nodal_data = st.StringSeries.read_array(
                    res_contents)

        return nodal_data, elemental_data

    def _parse_res(self, string_series, len_data, *, is_elemental=False):
        if string_series is None:
            return {}
        string_series = string_series.strip()
        component_num_end = 0
        for s in string_series:
            if re.search(r'^[\*a-zA-Z]', s):
                break
            component_num_end += 1
        component_nums = np.concatenate(
            [np.array(re.split(r'\s+', s), dtype=int)
             for s in string_series[0:component_num_end]])
        n_variables = len(component_nums)
        variable_names = string_series[
            component_num_end:component_num_end+n_variables]

        raw_data = string_series[component_num_end+n_variables:]
        stride = int(len(raw_data) / len_data)
        if stride * len_data != len(raw_data):
            raise ValueError(f"res file format not supported.")

        ids = raw_data[0::stride]
        data = st.StringSeries.connect_all(
            [raw_data[s::stride] for s in range(1, stride)], delimiter=' ')
        formatted_data = ids.connect(data, delimiter=' ')
        dict_fem_attribute = formatted_data.to_dict_fem_attributes(
                variable_names, component_nums, delimiter=' ')
        if is_elemental:
            return {
                fem_attribute.name:
                self.elements.generate_elemental_attribute(
                    fem_attribute.name, fem_attribute.ids, fem_attribute.data)
                for fem_attribute in dict_fem_attribute.values()}
        else:
            return dict_fem_attribute

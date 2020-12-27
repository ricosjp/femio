
import numpy as np

from ...fem_data import FEMData
from ...fem_elemental_attribute import FEMElementalAttribute
from ...util import string_parser as st


class UCDData(FEMData):
    """FEMEntity of AVS UCD version."""

    @classmethod
    def read_files(cls, file_names, read_mesh_only=False, time_series=False):
        """Initialize UCDEntity object.

        Args:
            file_names: list of str
                File names.
            read_mesh_only: bool, optional [False]
                If true, read mesh (nodes and elements) and ignore
                material data, results and so on.
        """
        obj = cls()
        obj.file_names = file_names

        if len(file_names) != 1:
            raise ValueError(
                f"{len(file_names)} files found. "
                'Specify file name by using read_files() instead of '
                'read_directory().')
        file_name = file_names[0]

        print('Parsing data')
        string_series = st.StringSeries.read_file(file_name).strip()
        headers = obj.read_headers(string_series)
        obj.nodes = obj.read_nodes(string_series, headers)
        obj.elements = obj.read_elements(string_series, headers)
        obj.settings['solution_type'] = None
        if read_mesh_only:
            return obj

        obj.nodal_data.update(obj.read_nodal_data(string_series, headers))
        obj.elemental_data.update(
            obj.read_elemental_data(string_series, headers))
        return obj

    def read_headers(self, string_series):
        """Read ucd file headers.

        Args:
            string_series: femio.util.string_parser.StringSeries
                StringSeries object of inp file.
        Returns:
            headers: dict
                Dictionary of header information e.g. the # of nodes.
        """
        top_header = string_series.iloc[[0]].to_values(
            r'\s+', data_type=int)[0]
        headers = {
            'n_node': top_header[0],
            'n_element': top_header[1],
            'all_dim_nodal_data': top_header[2],
            'all_dim_elemental_data': top_header[3],
        }

        if headers['all_dim_nodal_data'] != 0:
            nodal_data_header = string_series.iloc[[
                headers['n_node'] + headers['n_element'] + 1
            ]].to_values(r'\s+', data_type=int)[0]
            headers.update({
                'n_nodal_data': nodal_data_header[0],
                'nodal_data_dims': nodal_data_header[1:],
            })
        else:
            headers.update({
                'n_nodal_data': 0,
                'nodal_data_dims': [0],
            })

        if headers['all_dim_elemental_data'] != 0:
            elemental_data_header = string_series.iloc[[
                headers['n_node'] + headers['n_element'] + 1
                + headers['n_nodal_data']
                + min(1, headers['n_nodal_data']) * (headers['n_node'] + 1)
            ]].to_values(r'\s+', data_type=int)[0]
            headers.update({
                'n_elemental_data': elemental_data_header[0],
                'elemental_data_dims': elemental_data_header[1:],
            })
        else:
            headers.update({
                'n_elemental_data': 0,
                'elemental_data_dims': [0],
            })

        return headers

    def read_nodes(self, string_series, headers):
        start = 1
        end = headers['n_node'] + 1
        return string_series.iloc[start:end].to_fem_attribute(
            'NODE', 0, slice(1, None), delimiter=r'\s+')

    def read_elements(self, string_series, headers):
        start = headers['n_node'] + 1
        end = start + headers['n_element']
        types = string_series.iloc[start:end].split_vertical_all(
            delimiter=r'\s+')[2]
        if np.all(types == types[0]):
            return FEMElementalAttribute(
                'ELEMENT',
                string_series.iloc[start:end].to_fem_attribute(
                    'ELEMENT', 0, slice(3, None),
                    delimiter=r'\s+', data_type=int))
        else:
            dict_string = {t: [] for t in np.unique(types)}
            for s, type_ in zip(string_series[start:end], types):
                dict_string[type_].append(s)
            return FEMElementalAttribute(
                'ELEMENT', {
                    type_: st.StringSeries(string).to_fem_attribute(
                        'ELEMENT', 0, slice(3, None),
                        delimiter=r'\s+', data_type=int)
                    for type_, string in dict_string.items()})

    def read_nodal_data(self, string_series, headers):
        if headers['all_dim_nodal_data'] == 0:
            return {}

        name_start = headers['n_node'] + 1 + headers['n_element'] + 1
        name_end = name_start + headers['n_nodal_data']
        names, units = string_series.iloc[
            name_start:name_end].split_vertical(1)

        start = name_end
        end = start + headers['n_node']

        return self._read_associated_data(
            string_series.iloc[start:end],
            names, units, headers['nodal_data_dims'])

    def read_elemental_data(self, string_series, headers):
        if headers['all_dim_elemental_data'] == 0:
            return {}

        node_shift = min(1, headers['all_dim_nodal_data']) * (
            headers['n_nodal_data'] + headers['n_node'] + 1)
        name_start = headers['n_node'] + 1 + headers['n_element'] + 1 \
            + node_shift
        name_end = name_start + headers['n_elemental_data']
        names, units = string_series.iloc[
            name_start:name_end].split_vertical(1)

        start = name_end
        end = start + headers['n_element']

        return self._read_associated_data(
            string_series[start:end],
            names, units, headers['elemental_data_dims'])

    def _read_associated_data(self, string_series, names, units, dims):
        cum_dim = 1
        associated_data = {}
        for name, unit, dim in zip(names, units, dims):
            associated_data.update({
                name:
                string_series.to_fem_attribute(
                    name, 0, slice(cum_dim, cum_dim + dim),
                    delimiter=r'\s+', data_unit=unit)})
            cum_dim += dim
        return associated_data

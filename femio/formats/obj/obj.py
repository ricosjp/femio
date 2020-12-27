import re

import numpy as np

from ...fem_attribute import FEMAttribute
from ...fem_data import FEMData
from ...fem_elemental_attribute import FEMElementalAttribute
from ...util import string_parser as st


class ObjData(FEMData):
    """FEMEntity of Wavefront obj version."""

    DICT_LENGTH2TYPE = {
        3: 'tri',
        4: 'quad',
    }

    @classmethod
    def read_files(cls, file_names, read_mesh_only=False, time_series=False):
        """Initialize ObjEntity object.

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
        obj.nodes = obj.read_nodes(string_series)
        obj.elements = obj.read_elements(string_series)
        obj.settings['solution_type'] = None
        return obj

    def read_nodes(self, string_series):
        node_data = string_series.find_match(r'v\s+').split_vertical(
            0, r'\s+')[1].to_values(r'\s+')
        node_ids = np.arange(len(node_data), dtype=int) + 1
        return FEMAttribute('NODE', node_ids, node_data)

    def read_elements(self, string_series):
        str_element_data = string_series.find_match(r'f\s+').split_vertical(
            0, r'\s+')[1]
        element_data = np.array([
            [int(i) for i in re.split(r'\s+', s)]
            for s in str_element_data], dtype=object)
        element_ids = np.arange(len(element_data), dtype=int) + 1
        lengths = np.array([len(e) for e in element_data])
        unique_lengths = np.unique(lengths)

        elements = {
            self.DICT_LENGTH2TYPE[length]:
            self.extract_elements(element_ids, element_data, lengths, length)
            for length in unique_lengths}
        return FEMElementalAttribute('ELEMENT', elements)

    def extract_elements(self, element_ids, element_data, lengths, length):
        filter_ = lengths == length
        return FEMAttribute(
            self.DICT_LENGTH2TYPE[length],
            element_ids[filter_], np.stack(element_data[filter_]))

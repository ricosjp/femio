from io import StringIO
import os

import numpy as np
import pandas as pd

from .. import fem_attribute


class ListStringSeries():

    def __init__(self, list_string_series):
        self._list_string_series = list_string_series
        return

    def __len__(self):
        return len(self._list_string_series)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._list_string_series[key]
        elif isinstance(key, list):
            return [self[i] for i in key]
        else:
            raise ValueError(f"Unexpected key: {key}")

    def strip(self):
        return [s.strip() for s in self]

    def expand_include(self, pattern, base_name):
        return [s.expand_include(pattern, base_name) for s in self]


class StringSeries(pd.Series):

    def __init__(self, *args, **kw):
        if len(args) == 0 or len(args[0]) == 0:
            kw['dtype'] = object
        super().__init__(*args, **kw)

    @property
    def _constructor(self):
        return StringSeries

    @classmethod
    def read_file(cls, file_name, *, pattern_ignore=None):
        """Read file and convert to numpy string array.

        Args:
            file_name: String of file name.
            pattern_ignore: String to be used for ignore unecessary line
                e.g. comment.
        Returns:
            StringDataFrame object. Each component corresponds to each line of
            the input file.
        """
        print(f"Reading file: {file_name}")
        s = pd.read_csv(
            file_name, header=None, index_col=None, sep='@', dtype=str)[0]
        # sep='@' because don't want to separate
        if pattern_ignore is None:
            return cls(s)
        else:
            return cls(s).find_match(
                pattern_ignore, negative_match=True)

    @classmethod
    def read_files(cls, file_names, *, pattern_ignore=None, separate=False):
        """Read files.

        Args:
            file_names: Array of strings indicating file names.
            pattern_ignore: String to be used for ignore unecessary line
                e.g. comment.
            separate: bool
                If True, return separated contents, namely, ListStringSeries
                object.
        Returns:
            StringDataFrame object. Each component corresponds to each line of
            input files (contents are concatenated).
        """
        if separate:
            list_string_series = ListStringSeries([
                cls.read_file(file_name, pattern_ignore=pattern_ignore)
                for file_name in file_names])
            if len(list_string_series) == 1:
                return list_string_series[0]
            else:
                return list_string_series
        else:
            return cls(pd.concat([
                cls.read_file(file_name, pattern_ignore=pattern_ignore)
                for file_name in file_names]))

    @classmethod
    def read_array(cls, _array, *, delimiter=',', str_format=None):
        """Read array to make StringSeries object.

        Args:
            array: Ndarray or list of NDarray to make StringSeries object.
            delimiter: String indicating delimiter to connect components in
                a raw (default: ',').
            str_format: Format string to be passed to numpy.savetxt.
        Returns: StringSeries object after reading arrays.
        """
        array = np.asarray(_array)
        if str_format is None and 'float' in str(array.dtype):
            str_format = '%.8E'
        if len(array.shape) == 1:
            if str_format is None:
                return cls(array.astype(str))
            else:
                sio = StringIO()
                np.savetxt(sio, array, fmt=str_format)
                return cls(sio.getvalue().split('\n')[:-1])
        elif len(array.shape) == 2 and array.shape[1] == 1:
            if str_format is None:
                try:
                    converted_array = array.astype(str)
                    # Array can be converted to other types
                    return cls(converted_array[:, 0])
                except ValueError:
                    # Array is realy object
                    return cls(np.array([
                        '\n'.join(delimiter.join(a) for a in arr.astype(str))
                        for arr in array[:, 0]
                    ]))
            else:
                sio = StringIO()
                np.savetxt(sio, array[:, 0], fmt=str_format)
                return cls(sio.getvalue().split('\n')[:-1])
        elif len(array.shape) > 2:
            raise ValueError(f"Too high dimensions: {array.shape}")
        else:
            pass

        a0 = array[:, 0]
        if str_format is None:
            s = cls(a0.astype(str))
            for a in array[:, 1:].T:
                s = s.connect(a.astype(str))
        else:
            sio = StringIO()
            np.savetxt(sio, a0, fmt=str_format)
            s = cls(sio.getvalue().split('\n')[:-1])
            for a in array[:, 1:].T:
                sio = StringIO()
                np.savetxt(sio, a, fmt=str_format)
                s = s.connect(sio.getvalue().split('\n')[:-1])
        return s

    @classmethod
    def connect_all(cls, list_data, delimiter=',', str_format=None):
        if len(list_data) == 0:
            return cls()
        if str_format is None:
            str_format = [None] * len(list_data)
        elif isinstance(str_format, str):
            str_format = [str_format] * len(list_data)
        if len(list_data) != len(str_format):
            raise ValueError(
                'When str_format is list, the length should be'
                'the same as that of list_data'
                f"({len(str_format)} vs {len(list_data)})")

        s = cls.read_array(list_data[0], str_format=str_format[0])
        for d, f in zip(list_data[1:], str_format[1:]):
            s = s.connect(
                cls.read_array(d, str_format=f), delimiter=delimiter)
        return s

    @classmethod
    def concat(cls, list_data, axis=0):
        return cls(pd.concat(list_data, axis=axis))

    def to_header_data(self, pattern):
        matches = self.str.match(pattern).values
        headers = self[matches]
        match_indices = np.concatenate([np.where(matches)[0], [len(self)]])
        list_indices = [
            range(i1+1, i2) for i1, i2
            in zip(match_indices[:-1], match_indices[1:])]
        return HeaderData(headers, list_indices, data=self)
        # header_dict = {
        #     header: self[i1+1:i2] for header, i1, i2
        #     in zip(headers, match_indices[:-1], match_indices[1:])}
        # return HeaderData(header_dict)

    def strip(self):
        return self.str.strip()

    def extract_captures(self, pattern, *, convert_values=False):
        captures = self.str.extract(pattern, expand=False)
        captures = captures[~pd.isnull(captures)]
        if convert_values:
            return captures.values
        else:
            return captures

    def find_match(self, pattern, *, allow_multiple_matches=True,
                   convert_values=False, negative_match=False):
        """Find match to the specified pattern.

        Args:
            pattern: Pattern to be used for matching.
            allow_multiple_matches: True to accept several matches.
                (Default = True)
            convert_values: Bool, [True]
                Flag to convert StringSeries to values
        Returns:
            StringSeries or ndarray of matches.
        """
        if negative_match:
            match = self[~self.str.contains(pattern)]
        else:
            match = self[self.str.contains(pattern)]
        if not allow_multiple_matches and len(match) > 1:
            raise ValueError(f"{len(match)} matches found. Expected 1.")
        if convert_values:
            return match.values
        else:
            return match

    def expand_include(self, pattern, base_name):
        """Expand data like 'include' statement. Expanded data is concatenated
        at the end of the non-expanded data.

        Args:
            pattern: Pattern showing include statement. Include file should be
                captured with the first expression.
            base_name: Directory name of the include file location.
        Returns:
            StringSeries object after expansion.
        """
        captures = self.extract_captures(pattern)
        include_files = [os.path.join(base_name, c) for c in captures]
        if len(include_files) == 0:
            return self
        include_ss = StringSeries.read_files(include_files)
        return pd.concat([self, include_ss], ignore_index=True)

    def to_fem_attribute(self, name, id_column, slice_data_columns, *,
                         data_type=float, delimiter=',',
                         data_unit='unit_unknown', generate_id2index=False):
        """Generate FEMAttribute object with parsing the series.

        Args:
            name: String indicating name of the attribute.
            lines: Ndarray of strings contains data.
            id_column: Int indicating the column of ids.
            slice_data_columns: Slice object indicating the columns of data.
            data_type: Type of the data (default: float)
            delimiter: String of delimiter. (default: ',')
            data_unit: String indicating unit of data.
                (default: 'unit_unknown')
            generate_id2index: bool
                If True, generate pandas.DataFrame of IDs and indices.
        Returns:
            femio.FEMAttribute object.
        """
        df = self.str.split(delimiter, expand=True)
        ids = df.values[:, id_column].astype(float).astype(int)
        data = df.values[:, slice_data_columns].astype(data_type)
        return fem_attribute.FEMAttribute(
            name, ids, data, data_unit=data_unit,
            generate_id2index=generate_id2index)

    def to_values(
            self, delimiter=',', data_type=float, to_rank1=False,
            until_column=None):
        """Delimit StringLines object with the specified delimiter to output
        ndarray of the specified data_type.

        Args:
            delimiter: String of delimiter (default: ',').
            data_type: Type of output data (default: float).
            to_rank1: Boolean to control output (True: rank-1, False: rank-2,
                default: False)
            until_column: int, optional, [None]
                Read until the specified column.
        Returns:
            Ndarray of the specified data_type.
        """
        data = self.delimit(delimiter).astype(data_type)[:, :until_column]
        # except ValueError:
        #     raise ValueError(self)
        if to_rank1:
            return np.concatenate(data)
        else:
            return data

    def delimit(self, delimiter=','):
        """Delimit StringLines object with the specified delimiter to output
        rank-2 ndarray of strings.

        Args:
            delimiter: String of delimiter (default: ',').
        Returns:
            rank-2 ndarray of string.
        """
        return self.str.split(delimiter, expand=True).values

    def split_vertical(self, index_cut, delimiter=','):
        """Split StringSeries object vertically.

        Args:
            index_cut: Index (= start index of 2nd obj) to cut the StringLines.
        Return:
            2-tuple of DataFrame objects after splitting.
        """
        if len(self) == 0:
            return (pd.DataFrame([]), pd.DataFrame([]))
        if index_cut == 0:
            pattern = f"([^{delimiter}]*){delimiter}(.*)"
        else:
            pattern = \
                f"((?:[^{delimiter}]*{delimiter}){{{index_cut - 1}}}" \
                + f"[^{delimiter}]*){delimiter}(.*)"

        df_split = self.str.extract(pattern, expand=True)
        return (StringSeries(df_split[0]), StringSeries(df_split[1]))

    def split_vertical_all(self, delimiter=','):
        """Split StringSeries object vertically. Output will be n StringSeries
        objects.

        Args:
        Return:
            n-tuple of StringSeries objexts after splitting.
        """
        if len(self) == 0:
            return (StringSeries([]), )
        delimitted_data = self.delimit(delimiter)
        return [StringSeries(d.T) for d in delimitted_data.T]

    def connect(self, other, delimiter=','):
        """Connect two StringSeries objects with specified delimiter.
        Lengths of two objects should be the same.

        Args:
            other: Other StringSeries object to be connected.
            delimiter: String to appear at the connection.
        Return:
            StringSeries object after connection.
        """
        if len(other) == 0:
            return self
        if len(self) != len(other):
            raise ValueError('Dimension different: {} vs {}'.format(
                len(self), len(other)))
        return StringSeries(self.str.cat(
            StringSeries(other).values, sep=delimiter, join='left'))

    def indices_match_clusters(self, pattern, *, negative_match=False):
        """Make cluster of indices of matches. Cluster means a group with
        continuous indices.

        Args:
            pattern: Pattern to be used for matching.
        Returns:
            list of ndarrays containing indices of each cluster.
        """
        indices_matches = self.indices_matches(
            pattern, negative_match=negative_match)
        diff_ind = np.diff(indices_matches)
        separation_indices = [i + 1 for i, d in enumerate(diff_ind) if d > 1]
        start_indices = [0] + separation_indices
        stop_indices = separation_indices + [len(indices_matches)]
        return [indices_matches[i1:i2] for i1, i2
                in zip(start_indices, stop_indices)]

    def indices_matches(self, pattern, *, negative_match=False):
        """Return indices of matched lines.

        Args:
            pattern: Pattern to be used for matching.
        Returns:
            Ndarray of ints indicating indices of matched lines.
        """
        matches = self.astype(str).str.contains(pattern)
        if negative_match:
            matches = ~matches
        if np.all(~matches):
            raise ValueError('No match found for: {}'.format(pattern))
        return np.array(range(len(matches)))[matches]

    def to_dict_fem_attributes(self, names, component_nums,
                               data_units=None, delimiter=','):
        """Generate dict of FEMAttribute objects with parsing the lines.

        Args:
            names: List of strings indicating names of the attributes.
            component_nums: List of ints indicating # of components of each
                attributes.
            data_units: List of strings indicating unit of data.
                (default: 'unit_unknown')
        Returns:
            Dict with key = name, value = fem.FEMAttribute.
        """
        if data_units is None:
            data_units = ['unit_unknown' for _ in names]
        nums = np.concatenate([[0], np.cumsum(component_nums)]) + 1
        ranges = [range(n1, n2) for n1, n2 in zip(nums[:-1], nums[1:])]
        return {name: self.to_fem_attribute(
            name, 0, r, delimiter=delimiter, data_unit=unit)
            for name, r, unit in zip(names, ranges, data_units)}


class HeaderData():
    def __init__(self, headers, list_indices, data):
        if len(headers) != len(list_indices):
            raise ValueError(
                f"Length different: {len(headers)} vs {len(list_indices)}")
        self.dict = data
        self.headers = headers
        self.list_indices = np.array([
            np.array(indices) for indices in list_indices], dtype=object)
        self.data = data

    def extract_headers(self, key):
        return self.headers.find_match(key)

    def extract_data(self, key, *, concatenate=True):
        indices = self.headers.str.contains(key)
        if not np.any(indices):
            return StringSeries([])
        if concatenate:
            concatenated_indices = np.concatenate(
                self.list_indices[indices])
            return self.data.iloc[concatenated_indices]
        else:
            return [self.data.iloc[index]
                    for index in self.list_indices[indices]]

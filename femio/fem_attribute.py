
import numpy as np
import pandas as pd

from .time_series_dataframe import TimeSeriesDataFrame


class FEMAttribute():
    """Represents one attribute of FEM data such as node, element,
    displacement, ...

    Attributes
    ----------
    name: str
        String indicating the name of the attribute.
    ids: numpy.ndarray[int]
        Ndattay of ints indicating either IDs e.g. node ID, element ID.
    data: numpy.ndarray
        Ndarray of data content.
    data_unit: str
        String indicating unit of the data.
    time_series: bool
        If True, it indicates time series data.
    """

    @classmethod
    def load(cls, name, file_, **kwargs):
        """Load npz file to create FEMAttribute object.

        Parameters
        ----------
        name: str
            Name of the attribute.
        file_: file, str, or path.Path
            File or file name to which the data is saved.

        Returns
        -------
        FEMAttribute
        """
        dict_data = np.load(file_, allow_pickle=True)
        return cls.from_dict(name, dict_data, **kwargs)

    @classmethod
    def from_dict(cls, name, dict_data, **kwargs):
        """Create FEMAttribute object from the specified dict_data.

        Parameters
        ----------
        name: str
            Name of the attribute.
        dict_data: Dict[str, numpy.ndarray]
            Dict mapping from attribute (ID or data) name to its values.

        Returns
        -------
        FEMAttribute
        """
        if len(dict_data) != 2:
            raise ValueError(f"Unexpected data to load: {dict_data}")
        for k, v in dict_data.items():
            if 'ids' in k:
                ids_values = v
            elif 'data' in k:
                data_values = v
            else:
                raise ValueError(f"Unexpected key: {k}")
        return cls(name, ids=ids_values, data=data_values, **kwargs)

    def __init__(
            self, name, ids, data, *,
            data_unit='unit_unknown', silent=False, generate_id2index=False,
            time_series=False, original_shape=None, parent=None):
        """Initialize FEMAttribute object.

        Parameters
        ----------
        name: str
            String indicating name of the attribute.
        ids: List[int] or List[str]
            List of IDs of data e.g. node ID.
        data: numpy.ndarray
            Data corrsponding to the ids.
        data_unit: str, optional
            String indicating unit of data. The default is 'unit_unknown'
        silent: bool, optional [False]
            If True, print data name to create
        generate_id2index: bool, optional [False]
            If True, generate pandas.DataFrame to convert id to index.
        time_series: bool, optional [False]
            If True, consider the first index represents the temporal
            direction.
        original_shape: List[int], optional [None]
            The original shape of the data.
        parent: FEMAttribute, optional [None]
            If fed, update parent also.
        """
        self.name = name
        self.time_series = time_series
        self.original_shape = original_shape
        self.parent = parent

        if isinstance(data, pd.DataFrame):
            data = data.values
        if self.original_shape is None:
            data = np.atleast_1d(data)
        else:
            data = np.reshape(data, self.original_shape)

        ids = np.atleast_1d(ids)
        self._validate_data_length(ids, data)
        if not silent:
            print("Creating data: {}".format(name))

        self._data_frame, self.original_shape = self._generate_data_frame(
            ids, data)
        self._data = data
        self.data_unit = data_unit

        self.generate_id2index = generate_id2index
        if self.generate_id2index:
            self.id2index = pd.DataFrame(
                np.arange(len(self.ids)), index=self.ids)

        return

    def _validate_data_length(self, ids, data):
        if self.time_series:
            data_length = data.shape[1]
        else:
            data_length = len(data)

        if len(ids) != data_length:
            raise ValueError(
                'Lengths of IDs and data are different: '
                + f"{len(ids)} vs {data.shape} for {self.name}")
        return

    def _generate_data_frame(self, ids, data):
        data = np.asarray(data)
        if self.time_series:
            data_frame = TimeSeriesDataFrame([
                pd.DataFrame(data=d, index=ids) for d in data])
            original_shape = data.shape
        else:
            data_length = len(data)
            original_shape = data.shape
            data_frame = pd.DataFrame(
                data=np.reshape(data, (data_length, -1)), index=ids)

        # NOTE: To be determined the shape is at least 2D or not
        # if len(original_shape) == 1:
        #     original_shape = (original_shape[0], 1)  # At least 2D

        return data_frame, original_shape

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return ('Name: {}\n'.format(self.name)
                + '\n'.join(['{}: {} ({})'.format(i, d, self.data_unit)
                             for i, d in zip(self.ids, self.data)]))

    def __getitem__(self, key):
        return self.loc[key].values

    @property
    def ids(self):
        return self._data_frame.index.values

    @ids.setter
    def ids(self, value):
        self._data_frame.index = value
        return

    @property
    def iloc(self):
        return _Indexer(self, self._data_frame.iloc)

    @property
    def loc(self):
        return _Indexer(self, self._data_frame.loc)

    @property
    def data(self):
        return np.reshape(self._data, self.original_shape)

    @property
    def data_frame(self):
        return self._data_frame

    @property
    def values(self):
        return self.data

    @data.setter
    def data(self, value):
        self._validate_data_length(self.ids, value)
        self._data_frame, self.original_shape = self._generate_data_frame(
            ids=self.ids, data=value)
        self._data = value
        self._update_parent()
        return

    @data_frame.setter
    def data_frame(self, new_data_frame):
        if isinstance(new_data_frame, pd.DataFrame):
            self._data_frame = new_data_frame
            self._data = new_data_frame.values
        elif isinstance(new_data_frame, (list, tuple)):
            self._data_frame = new_data_frame
            self._data = np.array([d.values for d in new_data_frame])
        else:
            raise ValueError(
                f"Unsupported new_data_frame type for: {new_data_frame}")
        self._update_parent()
        return

    def _update_parent(self):
        if self.parent is None:
            return
        self.parent._data_frame.loc[self._data_frame.index] = self._data_frame
        return

    def update_data(self, values):
        self.data = values
        return

    def update(self, ids, values, *, allow_overwrite=False):
        """Update FEMAttribute with new ids and values.

        Parameters
        ----------
        ids: List[str], List[int], or int
            IDs of new rows.
        values: numpy.ndarray, float, or int
            Values of new rows.
        allow_overwrite: bool, optional
            If True, allow overwrite existing rows. The default is False.
        """
        if not isinstance(values, (np.ndarray, list, tuple)):
            values = [values]
        if not isinstance(ids, (np.ndarray, list, tuple)):
            ids = [ids]
        new_data_frame, new_shape = self._generate_data_frame(ids, values)

        if allow_overwrite:
            updated_data_frame = new_data_frame.combine_first(self._data_frame)
        else:
            # Append data if possible
            updated_data_frame = self._data_frame.append(
                new_data_frame, verify_integrity=True)
        if self.time_series:
            raise NotImplementedError
        else:
            new_length = len(updated_data_frame)
            self.original_shape = [new_length] + list(new_shape[1:])
        self.data_frame = updated_data_frame
        return

    def filter_with_ids(self, ids):
        return FEMAttribute(
            self.name, ids, self._data_frame.loc[ids], silent=True)

    def ids2indices(self, ids, id2index=None):
        """Return indices corresponding input ids.

        Parameters
        ----------
        ids: numpy.ndarray or femio.FEMElementalAttribute
            IDs.
        id2index: pandas.DataFrame, optional
            DataFrame of IDs and indices. If not fed and self does not have
            it, this function raises ValueError.

        Returns
        -------
        indices: numpy.ndarray
            Indices corresponding to ids.
        """
        if isinstance(ids, dict):
            ret = [
                self.ids2indices(v.data, id2index=id2index)
                for v in ids.values()]
            return ret
        if self.generate_id2index:
            id2index = self.id2index
        else:
            if id2index is None:
                raise ValueError('Must feed id2index')
        ids = np.asarray(ids)
        if str(ids.dtype) == 'object':
            try:
                ids = ids.astype(int)
                return self.ids2indices(ids, id2index=id2index)
            except (TypeError, ValueError):
                return [
                    self.ids2indices(ids_, id2index=id2index) for ids_ in ids]
        else:
            shape = ids.shape
            return np.reshape(
                np.ravel(id2index.loc[np.ravel(ids)].values), shape)

    def to_dict(self, prefix=None):
        """Convert to dict.

        Parameters
        ----------
        prefix: str, optional
            If fed, add f"{prefix}/" to the dictionary key.

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionay which maps f"{attribute_name}_ids" or
            f"{attribute_name}_data" to data contents.
        """
        if prefix is None:
            prefix = ''
        else:
            prefix = f"{prefix}/"
        return {f"{prefix}ids": self.ids, f"{prefix}data": self.data}

    def save(self, file_):
        """Save the contents.

        Parameters
        ----------
        file_: file, str, or path.Path
            File or file name to which the data is saved.

        Returns
        -------
        None
        """
        np.savez(file_, **self.to_dict())
        return


class _Indexer():

    def __init__(self, original_fem_attribute, indexer):
        """Manage iloc and loc access of the FEMAttribute object.

        Parameters
        ----------
        original_fem_attribute: FEMAttribute
            The original FEMAttribute object.
        indexer: pandas._iLocIndexer or pandas._LocIndexer
            Indexer object.
        """
        self.original_fem_attribute = original_fem_attribute
        self.indexer = indexer
        return

    def __getitem__(self, key):
        sliced_df = self.indexer[key]
        if sliced_df.ndim == 1:
            new_length = 1
            ids = [key]
        else:
            if self.original_fem_attribute.time_series:
                new_length = sliced_df.space_length
            else:
                new_length = len(sliced_df)
            ids = sliced_df.index

        original_shape = list(self.original_fem_attribute.original_shape)
        if self.original_fem_attribute.time_series:
            new_shape = [
                original_shape[0], new_length] + original_shape[2:]
        else:
            new_shape = [new_length] + original_shape[1:]

        return FEMAttribute(
            name=self.original_fem_attribute.name,
            ids=ids, data=sliced_df.values, silent=True,
            generate_id2index=self.original_fem_attribute.generate_id2index,
            time_series=self.original_fem_attribute.time_series,
            original_shape=new_shape, parent=self.original_fem_attribute)

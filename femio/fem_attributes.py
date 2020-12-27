from pathlib import Path

import numpy as np

from . import config
from .fem_attribute import FEMAttribute
from .fem_elemental_attribute import FEMElementalAttribute


class FEMAttributes:
    """Represents dictionary of FEMAttributes.

    Attributes
    ----------
    data: Dict[str, femio.FEMAttribute]
            or Dict[str, femio.FEMElementalAttribute]
    """

    @classmethod
    def load(cls, npz_file_name, **kwargs):
        """Load data from npz file.

        Parameters
        ----------
        npz_file: file, str, or pathlib.Path
            Npz file.

        Returns
        -------
        FEMAttributes
        """
        npz_file_name = Path(npz_file_name)
        if not npz_file_name.is_file():
            return cls({})

        dict_data = np.load(npz_file_name, allow_pickle=True)
        return cls.from_dict(dict_data, **kwargs)

    @classmethod
    def from_dict(cls, dict_data, **kwargs):
        """Create FEMAttributes object from the specified dict_data.

        Parameters
        ----------
        dict_data: Dict[str, numpy.ndarray]
            Dict mapping from attribute (ID or data) name to its values.

        Returns
        -------
        FEMAttribute
        """
        if 'is_elemental' in kwargs and kwargs['is_elemental']:
            attribute_class = FEMElementalAttribute
        else:
            attribute_class = FEMAttribute

        split_dict_data = cls._split_dict_data(dict_data)
        return cls([
            attribute_class.from_dict(k, v)
            for k, v in split_dict_data.items()], **kwargs)

    @classmethod
    def _split_dict_data(cls, dict_data):
        unique_attribute_names = np.unique([
            k.split('/')[0] for k in dict_data.keys()])
        return {
            unique_attribute_name:
            {
                k: v for k, v in dict_data.items()
                if unique_attribute_name == k.split('/')[0]}
            for unique_attribute_name in unique_attribute_names}

    @classmethod
    def from_meshio(cls, ids, dict_data, is_elemental=False):
        if is_elemental:
            elemental_data = {}
            for cell_type in dict_data.keys():
                for attribute_name, attribute_data in dict_data[
                        cell_type].items():
                    if attribute_name not in elemental_data:
                        elemental_data[attribute_name] = {}
                    elemental_data[attribute_name].update({
                        cell_type:
                        FEMAttribute(attribute_name, ids, attribute_data)})
            attributes = {
                attribute_name:
                FEMElementalAttribute(attribute_name, attribute_data)
                for attribute_name, attribute_data in elemental_data.items()}
            return cls(attributes, is_elemental=True)
        else:
            return cls({
                k: FEMAttribute(k, ids, v) for k, v in dict_data.items()})

    def __init__(
            self, attributes=None, names=None, ids=None, list_arrays=None, *,
            is_elemental=False):
        """Initialize FEMAttributes object.

        Parameters
        ----------
        attributes: List[femio.FEMAttribute] or Dict[str, femio.FEMAttribute],
                    optional
            List of FEMAttributes.
        names: List[str], optional
            Attribute names.
        ids: List[int] or List[str], optional
            List of IDs.
        list_arrays: List[numpy.ndarray], optional
            List of ndarray.
        is_elemental: bool, optional
            If True, create dict of FEMElementalAttributes instead of
            FEMAttributes. The default is False.
        """
        self.is_elemental = is_elemental
        if self.is_elemental:
            self.attribute_class = FEMElementalAttribute
        else:
            self.attribute_class = FEMAttribute

        if attributes is not None:
            if isinstance(attributes, dict):
                self.data = attributes
            else:
                self.data = {
                    attribute.name: attribute for attribute in attributes}
        elif ids is not None and list_arrays is not None:
            self.data = {
                name: self.attribute_class(name, ids=ids, data=data)
                for name, data in zip(names, list_arrays)}

        else:
            raise ValueError('Feed attributes or (names, ids, list_arrays).')

        self.material_overwritten = False
        return

    def __len__(self):
        return len(self.data)

    def _get_key(self, key):
        if key in self.keys():
            return key

        if key in config.DICT_ALIASES:
            return config.DICT_ALIASES[key]
        else:
            return key

    def __contains__(self, item):
        return self._get_key(item) in self.data

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[self._get_key(key)]
        else:
            return [self.data[self._get_key(k)] for k in key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.data[self._get_key(key)] = value
        else:
            for k in key:
                self.data[self._get_key(k)] = value
        return

    def __delitem__(self, key, value):
        self.pop(self._get_key(key))
        return

    def get_data_length(self):
        lengths = np.array([len(v) for v in self.values()])
        if np.all(lengths[0] == lengths):
            return lengths[0]
        else:
            raise ValueError('Data has different lengths')

    def get_attribute_ids(self, key, *, mandatory=True):
        """Get IDs of the specified attribute.

        Parameters
        ----------
        key: str or List[str]
            key to access the data.
        mandatory: bool, optional
            If True, raise ValueError if no data is found. The default is True.

        Returns
        -------
        data: numpy.ndarray or List[numpy.ndarray]
        """
        if isinstance(key, str):
            self._handle_key_missing(key, mandatory)
            return self[self._get_key(key)].ids
        else:
            for k in key:
                self._handle_key_missing(k, mandatory)
            return [d.ids for d in self[self._get_key(key)]]

    def get_attribute_data(self, key, *, mandatory=True):
        """Get contents of the specified attribute.

        Parameters
        ----------
        key: str or List[str]
            key to access the data.
        mandatory: bool, optional
            If True, raise ValueError if no data is found. The default is True.

        Returns
        -------
        data: numpy.ndarray or List[numpy.ndarray]
        """
        if isinstance(key, str):
            self._handle_key_missing(key, mandatory)
            return self[self._get_key(key)].data
        else:
            for k in key:
                self._handle_key_missing(k, mandatory)
            return [d.data for d in self[self._get_key(key)]]

    def set_attribute_data(
            self, key, data, *, allow_overwrite=False, name=None):
        """Set attribute data.

        Parameters
        ----------
        key: str
            Key of the new data.
        data: numpy.ndarray
            New data which has the same length as these of existing attributes.
        allow_overwrite: bool, optional
            If True, allow overwriting existing data. The default is False.
        name: str, optional
            The name of the new attribute. The default is the same as the key.
        """
        if not allow_overwrite and key in self.data:
            raise ValueError(
                f"Cannot overwrite the existing attribute: {key}.")
        if not self.are_same_lengths():
            raise ValueError(
                f"Attributes have various lengths. Specify IDs.")

        if name is None:
            name = key

        ids = list(self.data.values())[0].ids
        self[key] = self.attribute_class(name, ids=ids, data=data)
        return

    def are_same_lengths(self):
        """See if the attributes have the same lengths."""
        lengths = np.array([len(v.data) for v in self.data.values()])
        return np.all(lengths == lengths[0])

    def _handle_key_missing(self, key, mandatory):
        if self._get_key(key) not in self.data:
            if mandatory:
                raise ValueError(
                    f"{self._get_key(key)} not found in "
                    f"{self.data.keys()}")
            else:
                return None

    def reset(self):
        """Reset data contents.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.data = {}
        return

    def pop(self, key, default=None):
        """Pop data contents.

        Parameters
        ----------
        key: str or List[str]
            key to access the data.

        Returns
        -------
        data: numpy.ndarray or List[numpy.ndarray]
        """
        if isinstance(key, str):
            return self.data.pop(self._get_key(key), default)
        else:
            return [self.data.pop(self._get_key(k), default) for k in key]

    def to_dict(self):
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
        dict_data = {}
        for k, v in self.data.items():
            dict_data.update(v.to_dict(prefix=k))
        return dict_data

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
        if len(self) == 0:
            return

        np.savez(file_, **self.to_dict())
        return

    def update(self, dict_attributes):
        """Update FEMAttribute data with new dictionary.

        Parameters
        ----------
        dict_attributes: Dict[str, FEMAttribute] or FEMAttributes

        Returns
        -------
        None
        """
        if isinstance(dict_attributes, dict):
            self.data.update(dict_attributes)
        elif isinstance(dict_attributes, FEMAttributes):
            self.data.update(dict_attributes.data)
        else:
            raise ValueError(f"Unknown dict type for: {dict_attributes}")
        if self.has_material(dict_attributes):
            self.material_overwritten = True
        return

    def update_time_series(self, list_dict_attributes):
        """Update FEMAttribute data with new dictionary.

        Parameters
        ----------
        list_dict_attributes:
                List[Dict[str, FEMAttribute]] or List[FEMAttributes]

        Returns
        -------
        None
        """
        attribute_names = list(list_dict_attributes[0].keys())
        dict_attribute_ids = {
            name: list_dict_attributes[0][name].ids
            for name in attribute_names}
        dict_attribute_data = {
            name: self._extract_time_series_data(list_dict_attributes, name)
            for name in attribute_names}
        dict_attributes = {
            name: self.attribute_class(
                name, ids=dict_attribute_ids[name],
                data=dict_attribute_data[name], silent=True)
            for name in attribute_names}
        self.update(dict_attributes)
        return

    def _extract_time_series_data(self, list_dict_attributes, name):
        return np.stack([a[name].data for a in list_dict_attributes])

    def overwrite(self, name, data, *, ids=None):
        """Overwrite data.

        Paremeters
        ----------
        name: str
            Attribute name to be overwritten.
        data: numpy.ndarray
            New data to overwrite with.
        ids: numpy.ndarray
            IDs for new data.
        """
        if name not in self:
            raise ValueError(f"{name} not in the data {self.keys()}")
        if ids is None:
            self[name].data = data
        else:
            fem_attribute = FEMAttribute(name, ids=ids, data=data)
            self[name] = fem_attribute
        if name in config.LIST_MATERIALS:
            self.material_overwritten = True
        return

    def update_data(self, ids, data_dict, *, allow_overwrite=False):
        """Update data with new data_dict.

        Parameters
        ----------
        ids: List[str], List[int], str, or int
            IDs of FEMAttributes.
        data_dict: Dict[str, np.ndarray]
            Dictionary of data mapping from property names to property values.
        allow_overwrite: bool, optional
            If True, allow overwrite existing rows. The default is False.
        """
        for attribute_name, attribute_value in data_dict.items():
            if attribute_name in self:
                self.data[attribute_name].update(
                    ids, attribute_value, allow_overwrite=allow_overwrite)
            else:
                self.data[attribute_name] = FEMAttribute(
                    attribute_name, ids, attribute_value)
        if self.has_material(data_dict):
            self.material_overwritten = True
        return

    def get_n_material(self, fem_attributes=None):
        """Count the number of material properties contained in the
        fem_attributes.

        Parameters
        ----------
        fem_attributes: FEMAttributes, optional
            If not fed, self will be used.

        Returns
        -------
        has_material: bool
        """
        if fem_attributes is None:
            fem_attributes = self
        return np.sum(np.array([
            material_property_name in fem_attributes
            for material_property_name in config.LIST_MATERIALS]))

    def has_material(self, fem_attributes=None):
        """Check if fem_attributes have materials.

        Parameters
        ----------
        fem_attributes: FEMAttributes, optional
            If not fed, self will be used.

        Returns
        -------
        has_material: bool
        """
        return self.get_n_material(fem_attributes) > 0

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def extract_dict(self, ids):
        """Extract FEMAttributes data with IDs.

        Parameters
        ----------
        ids: List[str], List[int], str, or int
            IDs of FEMAttributes to extract.

        Returns
        -------
        extracted_dict: Dict[str, np.ndarray]
            Extracted dict mapping from attribute names to attribute values.
        """
        return {k: v.loc[ids].values for k, v in self.items()}

    def to_meshio(self):
        if self.is_elemental:
            cell_data = {}
            for attribute_name, attribute_data in self.items():
                for element_type, attribute in attribute_data.items():
                    if element_type not in cell_data:
                        cell_data[element_type] = {}
                    cell_data[element_type].update({
                        attribute_name: attribute.data})
            return cell_data
        else:
            return {
                attribute_name: attribute_data.data
                for attribute_name, attribute_data in self.items()}

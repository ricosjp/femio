
import numpy as np
import pandas as pd

from .fem_attribute import FEMAttribute
from . import config


class FEMElementalAttribute(dict):
    """Reperesents FEM element."""

    ELEMENT_TYPES = [
        'line',
        'line2',
        'spring',
        'tri',
        'tri2',
        'quad',
        'quad2',
        'tet',
        'tet2',
        'pyr',
        'pyr2',
        'prism',
        'prism2',
        'hex',
        'hex2',
    ]

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
        FEMElementalAttribute
        """
        dict_data = np.load(file_)
        return cls.from_dict(name, dict_data, **kwargs)

    @classmethod
    def from_dict(cls, name, dict_data, **kwargs):
        """Create FEMElementalAttribute object from the specified dict_data.

        Parameters
        ----------
        name: str
            Name of the attribute.
        dict_data: Dict[str, numpy.ndarray]
            Dict mapping from element type name to its values.

        Returns
        -------
        FEMAttribute
        """
        split_dict_data = cls._split_dict_data(dict_data)
        return cls(name, {
            element_type: FEMAttribute.from_dict(name, v)
            for element_type, v in split_dict_data.items()}, **kwargs)

    @classmethod
    def _split_dict_data(cls, dict_data):
        def _extract_element_type(string):
            split_strings = string.split('/')
            if len(split_strings) == 2:
                return split_strings[0]
            elif len(split_strings) == 3:
                return split_strings[1]
            else:
                raise ValueError(f"Unexpected string format: {string}")

        unique_element_types = np.unique([
            _extract_element_type(k) for k in dict_data.keys()])
        return {
            unique_element_type:
            {k: v for k, v in dict_data.items() if unique_element_type in k}
            for unique_element_type in unique_element_types}

    @classmethod
    def from_meshio(cls, cell_data):
        # NOTE: So far only support tetra10
        return FEMElementalAttribute('ELEMENT', {
            config.DICT_MESHIO_ELEMENT_TO_FEMIO_ELEMENT[k]:
            cls._from_meshio(k, v)
            for k, v in cell_data.items() if k in ['tetra', 'tetra10']})

    @classmethod
    def _from_meshio(cls, cell_type, data):
        if cell_type == 'tetra10':
            cell = cls._from_meshio_tet2(data)
        else:
            cell = data
        return FEMAttribute(cell_type, ids=np.arange(len(cell))+1, data=cell+1)

    @classmethod
    def _from_meshio_tet2(cls, data):
        return np.concatenate([
            data[:, :4],
            data[:, [5]], data[:, [6]], data[:, [4]],
            data[:, 7:]], axis=1)

    def __init__(
            self, name, data=None, *,
            ids=None, use_object=False, silent=False):
        """Create elements data from FEMAttribute object or dict of
        FEMAttribute objects.

        Parameters
        ----------
        elements: FEMAttribute or dict of FEMAttribute
            Input elements data
        ids: numpy.ndarray
            (n_element, )-shaped ndarray of element IDs.
        use_object: bool, optional [False]
            If True, use object for values.
        """
        if isinstance(data, FEMAttribute):
            element_type = self.detect_element_type(data.data)
            self.update({element_type: data})
        elif isinstance(data, FEMElementalAttribute):
            self.update(data)
        elif isinstance(data, dict):
            self.update(data)
        elif isinstance(data, np.ndarray):
            self.update({'unknown': FEMAttribute(name, ids=ids, data=data)})
        else:
            raise ValueError(f"Invalid input type: {data.__class__}")

        self.name = name
        if self.get_n_element_type() == 1:
            for k, v in self.items():
                self.ids = v.ids
                self.data = v.data
                self.element_type = k
                self.types = np.array([k] * len(self.ids))
        else:
            self.element_type = 'mix'
            ids = np.array([
                i
                for t in self.ELEMENT_TYPES if t in self
                for i in self[t].ids])
            data = np.array([
                d
                for t in self.ELEMENT_TYPES if t in self
                for d in self[t].data], dtype=object)
            types = np.array([
                t
                for t in self.ELEMENT_TYPES if t in self
                for _ in self[t].ids])
            sorted_indices = np.argsort(ids)

            self.ids = ids[sorted_indices]
            self.data = data[sorted_indices]
            self.types = types[sorted_indices]
            if len(np.unique(self.ids)) != len(self.data):
                raise ValueError('Element ID is not unique')

        self.unique_types = np.unique(self.types)
        self.id2index = pd.DataFrame(
            data=np.arange(len(self.ids)), index=self.ids)
        self.ids_types = pd.DataFrame(
            data=self.types, index=self.ids)
        self.dict_type_ids = {
            key: value.ids for key, value in self.items()}

        return

    def get_n_element_type(self):
        return len(self.keys())

    def __len__(self):
        return len(self.ids)

    def get_attribute_ids(self):
        return self.ids

    def to_first_order(self):
        return FEMElementalAttribute('ELEMENT', {
            element_type: FEMAttribute(
                element_type, elements.ids, self._to_first_order(
                    element_type, elements.data)
            ) for element_type, elements in self.items()}, ids=self.ids)

    def _to_first_order(self, element_type, element_data):
        if '2' not in element_type:
            return element_data

        if element_type == 'tet2':
            return element_data[:, :4]
        if element_type == 'hex2':
            return element_data[:, :8]
        else:
            raise ValueError(f"Unsupported type: {element_type}")

    def detect_element_type(self, element_data):
        n_node_per_element = element_data.shape[1]
        if n_node_per_element == 1:
            element_type = 'pt'
        elif n_node_per_element == 2:
            element_type = 'line'
        elif n_node_per_element == 3:
            element_type = 'tri'
        elif n_node_per_element == 4:
            # Colud be quad, but default to tet
            element_type = 'tet'
        elif n_node_per_element == 10:
            element_type = 'tet2'
        elif n_node_per_element == 8:
            element_type = 'hex'
        else:
            raise NotImplementedError(
                'Unsupported # of nodes per elements: '
                f"{self.elements.data.shape[1]}")
        return element_type

    def filter_with_ids(self, ids):
        indices = self.id2index.loc[ids].values[:, 0]
        filtered_element_data = self.data[indices]
        filtered_element_types = self.types[indices]
        existing_types = [
            t for t in self.ELEMENT_TYPES if t in filtered_element_types]
        return FEMElementalAttribute(self.name, {
            t: self._filter_with_type(
                t, ids, filtered_element_data, filtered_element_types)
            for t in existing_types})

    def _filter_with_type(
            self, type_, element_ids, element_data, element_types):
        filter_ = element_types == type_
        return FEMAttribute(
            type_, element_ids[filter_], np.stack(element_data[filter_]))

    def generate_elemental_attribute(self, name, ids, data):
        """Generate elemental attribute from IDs and data.

        Parameters
        ----------
        name: str
            The name of the attribute.
        ids: List[int] or List[str]
            Attribute IDs. All IDs should be in the element IDs.
        data: numpy.ndarray
            Attribute data.

        Returns
        -------
        elemental_attribute: FEMElementalAttribute
            Generated FEMElementalAttribute object.
        """
        dict_elemental_attribute = {}
        data_frame = pd.DataFrame(data, index=ids)
        for type_, type_ids in self.dict_type_ids.items():
            intersect_ids = np.intersect1d(type_ids, ids)
            if len(intersect_ids) == 0:
                continue
            intersect_data = data_frame.loc[intersect_ids]
            dict_elemental_attribute.update({
                type_:
                FEMAttribute(
                    name, ids=intersect_ids, data=intersect_data.values)})

        return FEMElementalAttribute(name, dict_elemental_attribute)

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

        dict_data = {}
        for key, value in self.items():
            dict_data.update(value.to_dict(prefix=f"{prefix}{key}"))
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

    def to_meshio(self, nodes):
        """Convert to meshio-like cell data. It assumes self is elements
        (not elemental_data).

        Parameters
        ----------
        nodes: FEMAttribute
            Node data.

        Returns
        -------
        cell_info: dict[str, numpy.ndarray]
            Dict mapping from cell type to cell connectivity data.
        """
        tmp_elements = FEMElementalAttribute('ELEMENT', {
            k:
            FEMAttribute(
                k, v.ids,
                self._to_meshio(k, v))
            for k, v in self.items()})
        return {
            config.DICT_FEMIO_ELEMENT_TO_MESHIO_ELEMENT[k]: v
            for k, v
            in tmp_elements._to_indices(nodes).items()}

    def _to_meshio(self, cell_type, element):
        if cell_type == 'tet2':
            return self._to_meshio_tet2(element.data)
        else:
            return element.data

    def _to_meshio_tet2(self, data):
        return np.concatenate([
            data[:, :4],
            data[:, [6]], data[:, [4]], data[:, [5]],
            data[:, 7:]], axis=1)

    def _to_indices(self, nodes):
        return {
            element_type: nodes.ids2indices(element_data.data)
            for element_type, element_data in self.items()}

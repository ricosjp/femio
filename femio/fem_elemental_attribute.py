
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
        'polygon',
        'tet',
        'tet2',
        'pyr',
        'pyr2',
        'prism',
        'prism2',
        'hex',
        'hex2',
        'hexprism',
        'polyhedron',
        'unknown',
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
        dict_data = np.load(file_, allow_pickle=True)
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
            element_type: FEMAttribute.from_dict(name, v, **kwargs)
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
        return FEMElementalAttribute('ELEMENT', {
            config.DICT_MESHIO_ELEMENT_TO_FEMIO_ELEMENT[k]:
            cls._from_meshio(k, v)
            for k, v in cell_data.items()
            if k in [
                'line', 'quad', 'triangle',
                'tetra', 'tetra10', 'hexahedron', 'wedge',
                'pyramid', 'hexa_prism']})

    @classmethod
    def _from_meshio(cls, cell_type, data):
        if cell_type == 'tetra10':
            cell = cls._from_meshio_tet2(data)
        else:
            cell = data
        return FEMAttribute(
            config.DICT_MESHIO_ELEMENT_TO_FEMIO_ELEMENT[cell_type],
            ids=np.arange(len(cell)) + 1, data=cell + 1)

    @classmethod
    def _from_meshio_tet2(cls, data):
        return np.concatenate([
            data[:, :4],
            data[:, [5]], data[:, [6]], data[:, [4]],
            data[:, 7:]], axis=1)

    def __init__(
            self, name, data=None, *,
            ids=None, use_object=False, silent=False, time_series=False,
            element_type=None, **kwargs):
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
        time_series: bool, optional [False]
            If True, consider the first index represents the temporal
            direction.
        element_type: str, optional [None]
            The type of element. Is is used when the input data is np.array.
        """
        self.time_series = time_series
        if isinstance(data, FEMAttribute):
            if element_type is None:
                element_type = self.detect_element_type(data.data)
            self.update({element_type: data})
        elif isinstance(data, FEMElementalAttribute):
            self.update(data)
        elif isinstance(data, dict):
            data = self._validate_keys(data)
            self.update(data)
        elif isinstance(data, np.ndarray):
            if ids is None:
                ids = np.arange(len(data)) + 1
            if element_type is None:
                element_type = 'unknown'
            self.update({element_type: FEMAttribute(
                name, ids=ids, data=data, time_series=self.time_series)})
        else:
            raise ValueError(f"Invalid input type: {data.__class__}")

        self.name = name
        self._update_self()

        return

    def _validate_keys(self, dict_data):
        for k in dict_data.keys():
            if k not in self.ELEMENT_TYPES:
                if len(dict_data) > 1:
                    raise ValueError(f"Unsupported element type: {k}")
                else:
                    return {'unknown': list(dict_data.values())[0]}
        return dict_data

    def _update_self(self):
        if self.get_n_element_type() == 1:
            for k, v in self.items():
                self._ids = v.ids
                self._data = v.data
                self._element_type = k
                self._types = np.array([k] * len(self.ids))
        else:
            self._element_type = 'mix'
            ids = np.array([
                i for t in self.keys() for i in self[t].ids])
            data = np.array([
                d for t in self.keys()
                for d in self[t].data], dtype=object)
            types = np.array([
                t
                for t in self.keys() for _ in self[t].ids])
            sorted_indices = np.argsort(ids)

            self._ids = ids[sorted_indices]
            self._data = data[sorted_indices]
            self._types = types[sorted_indices]
            if len(np.unique(self.ids)) != len(self.data):
                print('Making element IDs unique')
                self._unique_element_ids()
                self._update_self()
                return

        self._unique_types = np.unique(self.types)
        self._id2index = pd.DataFrame(
            data=np.arange(len(self.ids)), index=self.ids)
        self._ids_types = pd.DataFrame(
            data=self.types, index=self.ids)
        self._dict_type_ids = {
            key: value.ids for key, value in self.items()}

        return

    def keys(self):
        return [t for t in self.ELEMENT_TYPES if t in self]

    def values(self):
        return [self[t] for t in self.ELEMENT_TYPES if t in self]

    def items(self):
        return [(t, self[t]) for t in self.ELEMENT_TYPES if t in self]

    def _unique_element_ids(self):
        offset = 0
        for element_type in self.keys():
            self[element_type].ids += offset
            offset += len(self[element_type])
        return

    @property
    def ids(self):
        return self._ids

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self.get_n_element_type() == 1:
            self[list(self.keys())[0]].data = value
        else:
            raise NotImplementedError
        self._update_self()
        return

    @property
    def element_type(self):
        return self._element_type

    @property
    def types(self):
        return self._types

    @property
    def unique_types(self):
        return self._unique_types

    @property
    def id2index(self):
        return self._id2index

    @property
    def ids_types(self):
        return self._ids_types

    @property
    def dict_type_ids(self):
        return self._dict_type_ids

    def to_vtk(self, nodes):
        return np.concatenate([
            np.concatenate([[len(e)], nodes.ids2indices(e)])
            for e in self.data])

    def update(self, *args, **kwargs):
        if isinstance(args[0], dict):
            dict_data = self._validate_keys(args[0])
            if len(args) > 1:
                super().update(dict_data, *args[1:], **kwargs)
            else:
                super().update(dict_data, **kwargs)
            self._update_self()
        else:
            self._update(*args, **kwargs)
        return

    def _update(self, ids, values, *, allow_overwrite=False):
        """Update FEMElementalAttribute with new ids and values.

        Parameters
        ----------
        ids: List[str], List[int], or int
            IDs of new rows.
        values: numpy.ndarray, float, or int
            Values of new rows.
        allow_overwrite: bool, optional
            If True, allow overwrite existing rows. The default is False.
        """
        if self.get_n_element_type() == 1:
            self[list(self.keys())[0]].update(
                ids, values, allow_overwrite=allow_overwrite)
        else:
            raise NotImplementedError
        self._update_self()
        return

    def get_n_element_type(self):
        return len(self.keys())

    def __len__(self):
        return len(self.ids)

    def _infer_type(self, data):
        n_node_per_element = data.shape[-1]
        if n_node_per_element == 1:
            return 'pt'
        elif n_node_per_element == 2:
            return 'line'
        elif n_node_per_element == 3:
            return 'tri'
        elif n_node_per_element == 4:
            raise ValueError(
                'When # of nodes per element is 4, explicitly input '
                'element_type.')
        elif n_node_per_element == 8:
            return 'hex'
        elif n_node_per_element == 10:
            return 'tet2'
        else:
            raise ValueError(
                f"Unexpected # of nodes per element: {n_node_per_element}")

    def get_attribute_ids(self):
        return self.ids

    def to_first_order(self):
        if self.is_first_order():
            return self
        else:
            return FEMElementalAttribute('ELEMENT', {
                element_type: FEMAttribute(
                    element_type, elements.ids, self._to_first_order(
                        element_type, elements.data)
                ) for element_type, elements in self.items()}, ids=self.ids)

    def is_first_order(self):
        return not np.any(['2' in t for t in self.unique_types])

    def _to_first_order(self, element_type, element_data):
        if '2' not in element_type:
            return element_data

        if element_type == 'tet2':
            return element_data[:, :4]
        if element_type == 'hex2':
            return element_data[:, :8]
        else:
            raise ValueError(f"Unsupported type: {element_type}")

    def to_surface(self, surface_ids):
        """Convert the FEMElementalAttribute object to surface.

        Parameters
        ----------
        surface_ids: numpy.ndarray
            [n_facet, n_node_per_facet]-shaped array of surface IDs.

        Returns
        -------
        FEMElementalAttribute:
            FEMElementalAttribute object of the surface.
        """
        if isinstance(surface_ids, dict):
            surface_ids_tuple = self._generate_surface_ids_tuple(surface_ids)
            surfaces = self._generate_surface(surface_ids_tuple)
        elif isinstance(surface_ids, tuple):
            surface_ids_tuple = surface_ids
            surfaces = self._generate_surface(surface_ids_tuple)
        else:
            surfaces = self._generate_surface_core(surface_ids)

        return FEMElementalAttribute('ELEMENT', surfaces)

    def _generate_surface_ids_tuple(self, surface_ids_dict):
        group_dict = {}
        for surface_ids in surface_ids_dict.values():
            if isinstance(surface_ids, tuple):
                for si in surface_ids:
                    size = si.shape[-1]
                    if group_dict.get(size, None) is None:
                        group_dict[size] = si
                    else:
                        group_dict[size] = np.concatenate(
                            [group_dict[size], si])
            else:
                shape = surface_ids.shape
                if len(shape) == 1:
                    if 'polygon' in group_dict:
                        group_dict['polygon'] = np.concatenate(
                            [group_dict['polygon'], surface_ids])
                    else:
                        group_dict['polygon'] = surface_ids
                else:
                    size = surface_ids.shape[-1]
                    if size in group_dict:
                        group_dict[size] = np.concatenate(
                            [group_dict[size], surface_ids])
                    else:
                        group_dict[size] = surface_ids
        return tuple(group_dict.values())

    def _generate_surface(self, tuple_surface_ids):
        ret = {}
        last_id_end = 0
        for surface_ids in tuple_surface_ids:
            if len(surface_ids) == 0:
                continue
            item_surface = self._generate_surface_core(
                surface_ids, last_id_end)
            key = list(item_surface.keys())[0]
            new_value = list(item_surface.values())[0]
            if key in ret:
                original_value = ret[key]
                try:
                    data = np.concatenate(
                        [original_value.data, new_value.data])
                except ValueError:
                    # Manage if the arrays are not in the same shape
                    data = np.array(
                        list(original_value.data) + list(new_value.data))
                ret[key] = FEMAttribute(
                    key,
                    ids=np.concatenate([original_value.ids, new_value.ids]),
                    data=data)
            else:
                ret.update(item_surface)
            last_id_end = new_value.ids[-1]
        return ret

    def _generate_surface_core(self, surface_ids, last_id_end=0):
        shape = surface_ids.shape
        if len(shape) == 1:
            element_type = 'polygon'
        else:
            n_node_per_element = shape[1]
            if n_node_per_element == 3:
                element_type = 'tri'
            elif n_node_per_element == 4:
                element_type = 'quad'
            elif n_node_per_element > 4:
                element_type = 'polygon'
            else:
                raise NotImplementedError(
                    f"Unsupported shape of elements: {shape}")
        return {
            element_type:
            FEMAttribute(
                element_type,
                np.arange(len(surface_ids)) + last_id_end + 1, surface_ids)}

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
        ids = ids[np.isin(ids, self.id2index.index)]
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
        if type_ == 'polyhedron':
            data = element_data[filter_]
        else:
            data = np.stack(element_data[filter_])
        return FEMAttribute(
            type_, element_ids[filter_], data)

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

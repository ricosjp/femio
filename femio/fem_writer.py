
import numpy as np

from .fem_attribute import FEMAttribute
from .fem_elemental_attribute import FEMElementalAttribute


class FEMWriter():

    def __init__(
            self, fem_data, *, include_displacement=False, overwrite=False):
        """Initialize FEMWriter object.

        Parameters
        ----------
        fem_data: FEMData
            FEMData object to write.
        """
        self.fem_data = fem_data
        return

    def try_convert_to_2d(self, mode='nodal'):
        if mode == 'nodal':
            fem_attributes = self.fem_data.nodal_data
            ids = self.fem_data.nodes.ids
        elif mode == 'elemental':
            fem_attributes = self.fem_data.elemental_data
            ids = self.fem_data.elements.ids
        else:
            raise ValueError(f"Unexpected mode: {mode}")
        fem_attributes.update(
            self._generate_time_series(fem_attributes, ids, mode=mode))
        return {
            key: value
            for key, value in fem_attributes.items()
            if len(value.data.shape) == 2
            and self._extract_dtype(value.data) != np.dtype('O')}

    def _generate_time_series(self, fem_attributes, ids, mode):
        n = len(ids)
        if mode == 'nodal':
            fem_attribute_class = FEMAttribute
        elif mode == 'elemental':
            fem_attribute_class = FEMElementalAttribute
        ret = {}
        for k, v in fem_attributes.items():
            if len(v.data.shape) == 3 and v.data.shape[1] == n:
                ret.update({
                    f"{k}_{i}":
                    fem_attribute_class(name=f"{k}_{i}", ids=ids, data=d)
                    for i, d in enumerate(v.data)})
        return ret

    def _extract_dtype(self, array):
        if hasattr(array, 'dtype'):
            dtype = array.dtype
        else:
            dtype = type(array)
        return dtype

    def _extract_first_order_element(self, element, element_type):
        if element_type[-1] != '2':
            return element.data, element_type
        else:
            if element_type == 'tet2':
                element = element.data[:, :4]
            else:
                raise ValueError(
                    f"Unknown element type: {element_type}")
        return element, element_type[:-1]

    def _convert_objectdict2arraydict(self, object_dict):
        return_dict = {}
        for key, value in object_dict.items():
            converted_value = self._convert_object2array(value.data)
            if len(converted_value.shape) == 2:
                return_dict.update({key: converted_value})
        return return_dict

    def _convert_object2array(self, objectarray):
        if objectarray.dtype != np.dtype('O'):
            return objectarray

        if hasattr(objectarray[0, 0], 'dtype'):
            original_dtype = objectarray[0, 0].dtype
        else:
            return objectarray

        row, col = objectarray.shape
        feature = objectarray[0, 0]
        stripped = np.stack([
            d.astype(original_dtype) for d in np.ravel(objectarray)])
        if len(feature.shape) == 0:
            return np.reshape(stripped, (row, col))
        else:
            return np.reshape(stripped, [row, col] + list(feature.shape))

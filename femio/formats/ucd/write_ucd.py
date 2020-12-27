
import numpy as np
import pandas as pd


class UCDWriter():

    def __init__(self, fem_data):
        self.fem_data = fem_data

    def write(self, file_name=None, *, overwrite=False):
        """Write FEM data in inp format.

        Args:
            fem_data: FEMData object to be output.
            file_name: File name of the output file. If not fed,
                input_filename.out.ext will be the output file name.
            overwrite: Bool, if True, allow averwrite files (Default: False.)
        """

        n_node = len(self.fem_data.nodes.ids)
        n_element = len(self.fem_data.elements.ids)

        # NOTE: So far write only non time series data whose shape == 2
        nodal_data_dict_2d = {
            key: value
            for key, value in self.fem_data.nodal_data.items()
            if len(value.data.shape) == 2
            and self._extract_dtype(value.data) != np.dtype('O')}
        nodal_data_dimensions = [
            v.data.shape[1] for v in nodal_data_dict_2d.values()]
        elemental_data_dict_2d = self._convert_objectdict2arraydict(
            self.fem_data.elemental_data)
        elemental_data_dimensions = [
            v.data.shape[1] for v in elemental_data_dict_2d.values()]

        with open(file_name, 'w') as f:
            # Header
            f.write(
                f"{n_node} {n_element} {int(np.sum(nodal_data_dimensions))}"
                f" {int(np.sum(elemental_data_dimensions))} 0\n")

            # Node
            f.write(pd.DataFrame(
                index=self.fem_data.nodes.ids, data=self.fem_data.nodes.data
            ).to_csv(sep=' ', header=False, na_rep='NaN'))

            # Element
            for element_type in self.fem_data.elements.ELEMENT_TYPES:
                if element_type not in self.fem_data.elements:
                    continue
                element = self.fem_data.elements[element_type]
                n_element = len(element.ids)
                first_element, first_element_type = \
                    self._extract_first_order_element(element, element_type)
                f.write(pd.DataFrame(
                    index=element.ids,
                    data=np.concatenate([
                        np.ones([n_element, 1], dtype=int),
                        np.array([[first_element_type] * n_element]).T,
                        first_element
                    ], axis=1)
                ).to_csv(sep=' ', header=False, na_rep='NaN'))

            # Nodal data
            n_nodal_data = len(nodal_data_dict_2d)
            if n_nodal_data > 0:
                f.write(f"{n_nodal_data} " + ' '.join(
                    str(d) for d in nodal_data_dimensions) + '\n')
                f.write(
                    '\n'.join(
                        f"{k}, unit_unknown" for k
                        in nodal_data_dict_2d.keys()) + '\n')
                f.write(pd.DataFrame(
                    index=self.fem_data.nodes.ids,
                    data=np.concatenate([
                        v.data for v
                        in nodal_data_dict_2d.values()], axis=1)
                ).to_csv(sep=' ', header=False, na_rep='NaN'))

            # Elemental data
            n_elemental_data = len(elemental_data_dict_2d)
            if len(elemental_data_dict_2d) > 0:
                f.write(f"{n_elemental_data} " + ' '.join(
                    str(d) for d in elemental_data_dimensions) + '\n')
                f.write(
                    '\n'.join(
                        f"{k}, unit_unknown" for k
                        in elemental_data_dict_2d.keys()) + '\n')
                f.write(pd.DataFrame(
                    index=self.fem_data.elements.ids,
                    data=np.concatenate([
                        v.data for v
                        in elemental_data_dict_2d.values()
                        if len(v.data.shape) == 2], axis=1)
                ).to_csv(sep=' ', header=False, na_rep='NaN'))

        return file_name

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

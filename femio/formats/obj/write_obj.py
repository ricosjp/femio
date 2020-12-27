
import pandas as pd


class OBJWriter():

    def __init__(self, fem_data):
        self.fem_data = fem_data

    def write(self, file_name=None, *, overwrite=False):
        """Write FEM data in obj format.

        Args:
            fem_data: FEMData object to be output.
            file_name: File name of the output file. If not fed,
                input_filename.out.ext will be the output file name.
            overwrite: Bool, if True, allow averwrite files (Default: False.)
        """

        with open(file_name, 'w') as f:
            # Node
            n_node = len(self.fem_data.nodes.ids)
            f.write(pd.DataFrame(
                index=['v']*n_node, data=self.fem_data.nodes.data
            ).to_csv(sep=' ', header=False, na_rep='NaN'))

            surface_indices, _ \
                = self.fem_data.extract_surface()
            if isinstance(surface_indices, dict):
                # Mixed elements
                for v in surface_indices.values():
                    n_element = len(v)
                    f.write(pd.DataFrame(
                        index=['f']*n_element, data=v+1
                    ).to_csv(sep=' ', header=False, na_rep='NaN'))
            else:
                n_element = len(surface_indices)
                f.write(pd.DataFrame(
                    index=['f']*n_element, data=surface_indices+1
                ).to_csv(sep=' ', header=False, na_rep='NaN'))


import pathlib

from tvtk.api import tvtk

from ...fem_data import FEMData
from ... import io


class EnsightGoldData(FEMData):
    """FEMData of Ensight Gold."""

    @classmethod
    def read_files(cls, file_names, read_mesh_only=False, time_series=False):
        """Initialize PolyVTKData object.

        Parameters
        ----------
        file_names: list of str
            File names.
        read_mesh_only: bool, optional
            If true, read mesh (nodes and elements) and ignore
            material data, results and so on. The default is False.
        """
        if isinstance(file_names, str):
            file_name = pathlib.Path(file_names)
        else:
            file_name = pathlib.Path(file_names[0])

        if not file_name.is_file():
            raise ValueError(f"{file_name} not found")

        reader = tvtk.EnSightGoldBinaryReader()
        reader.case_file_name = file_name.name
        reader.file_path = str(file_name.parent)
        reader.update()
        blocks = reader.get_output()

        return [
            io.convert_vtk_unstructured_to_femio(blocks.get_block(i))
            for i in range(blocks.trait_get('number_of_blocks')[
                'number_of_blocks'])]

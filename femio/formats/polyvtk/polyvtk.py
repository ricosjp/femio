
import pathlib

from tvtk.api import tvtk

from ...fem_data import FEMData
from ... import io


class PolyVTKData(FEMData):
    """FEMData of VTK with polyhedron."""

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
            file_name = file_names
        else:
            file_name = file_names[0]

        if not pathlib.Path(file_name).is_file():
            raise ValueError(f"{file_name} not found")

        reader = tvtk.XMLUnstructuredGridReader(file_name=file_name)
        reader.update()
        mesh = reader.get_output()

        return io.convert_vtk_unstructured_to_femio(mesh)

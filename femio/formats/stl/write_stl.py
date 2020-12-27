
import numpy as np
import stl


class STLWriter():

    def __init__(self, fem_data, *, include_displacement=False):
        """Initialize STLWrite object.

        Args:
            fem_data: FEMData
            include_displacement: bool, optional [False]
                If True, create STL with node + displacement instead of node
                only.
        """
        self.fem_data = fem_data
        surface_indices, _ = \
            self.fem_data.extract_surface()
        if include_displacement:
            points = self.fem_data.nodes.data + self.fem_data.access_attribute(
                'displacement')
        else:
            points = self.fem_data.nodes.data
        self.surface_points = np.array([
            [points[facet[0]], points[facet[1]],
             points[facet[2]]]
            for facet in surface_indices])

    def write(self, file_name=None, *, overwrite=False):
        """Write FEM data in STL format.

        Args:
            file_name: str
                File name of the output file. If not fed,
                input_filename.out.stl will be the output file name.
            overwrite: bool, optional [False]
                If True, allow averwrite files.
        """
        empty_data = np.zeros(len(self.surface_points),
                              dtype=stl.base.BaseMesh)
        mesh = stl.mesh.Mesh(empty_data, remove_empty_areas=False)
        mesh.vectors = self.surface_points
        mesh.save(file_name)

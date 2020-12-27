import functools

import numpy as np

from .fem_attribute import FEMAttribute


class GeometryProcessorMixin:

    @functools.lru_cache(maxsize=1)
    def calculate_element_areas(
            self, *, raise_negative_area=False, return_abs_area=True):
        """Calculate areas of each element assuming that the geometry of
        higher order elements is the same as that of order 1 elements.
        Calculated areas are returned and also stored in
        the fem_data.elemental_data dict with key = 'area' .

        Args:
            raise_negative_area: bool, optional [False]
                If True, raise ValueError when negative area exists.
            return_abs_area: bool, optional [True]
                If True, return absolute area instead of signed area.
        Returns:
            areas: numpy.ndarray
        """
        if self.elements.element_type in ['tri']:
            areas = self._calculate_element_areas_tri()
        elif self.elements.element_type in ['quad']:
            areas = self._calculate_element_areas_quad()
        else:
            raise NotImplementedError

        # Handle negative volumes according to the settings
        if raise_negative_area and np.any(areas < 0.):
            raise ValueError('Negative area found.')
        if return_abs_area:
            areas = np.abs(areas)

        self.elemental_data.update({
            'area': FEMAttribute('Area', self.elements.ids, areas)})
        return areas

    def _calculate_element_areas_tri(self):
        crosses = self._calculate_tri_crosses()
        return np.linalg.norm(crosses, axis=1, keepdims=True) / 2.

    def _calculate_element_areas_quad(self):
        elements = self.elements.data[:, :4]
        node0_points = self.collect_node_positions_by_ids(elements[:, 0])
        node1_points = self.collect_node_positions_by_ids(elements[:, 1])
        node2_points = self.collect_node_positions_by_ids(elements[:, 2])
        node3_points = self.collect_node_positions_by_ids(elements[:, 3])

        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        crosses1 = np.cross(v10, v20)
        v32 = node3_points - node2_points
        v02 = node0_points - node2_points
        crosses2 = np.cross(v32, v02)
        areas1 = np.linalg.norm(crosses1, axis=1, keepdims=True) / 2
        areas2 = np.linalg.norm(crosses2, axis=1, keepdims=True) / 2
        return areas1 + areas2

    @functools.lru_cache(maxsize=1)
    def calculate_edge_lengths(self):
        """Calculate edge lengths of each element.
        Calculated lengths are returned and also stored in
        the fem_data.elemental_data dict with key = 'edge_lengths' .

        Returns:
            edge_lengths: numpy.ndarray
        """
        if self.elements.element_type in ['tri', 'quad']:
            edge_lengths = self._calculate_edge_lengths_polygon()
        else:
            raise NotImplementedError

        self.elemental_data.update({
            'edge_lengths':
            FEMAttribute('EdgeLengths', self.elements.ids, edge_lengths)})
        return edge_lengths

    def _calculate_edge_lengths_polygon(self):
        elements = self.elements.data
        n = elements.shape[1]
        points = [
            self.collect_node_positions_by_ids(elements[:, i])
            for i in range(n)]
        edge_lengths = np.empty(elements.shape, dtype=np.float64)

        for i in range(n):
            v = points[(i + 1) % n] - points[i]
            edge_lengths[:, i] = np.linalg.norm(v, axis=1)

        return edge_lengths

    @functools.lru_cache(maxsize=1)
    def calculate_angles(self):
        """Calculate angles of each element.
        Calculated angles are returned and also stored in
        the fem_data.elemental_data dict with key = 'angles' .

        Returns:
            angles: numpy.ndarray
        """
        if self.elements.element_type in ['tri', 'quad']:
            angles = self._calculate_angles_polygon()
        else:
            raise NotImplementedError

        self.elemental_data.update({
            'angles': FEMAttribute('Angles', self.elements.ids, angles)})
        return angles

    def _calculate_angles_polygon(self):
        elements = self.elements.data
        n = elements.shape[1]
        points = [
            self.collect_node_positions_by_ids(elements[:, i])
            for i in range(n)]
        angles = np.empty(elements.shape, dtype=np.float64)

        for i in range(n):
            v1 = points[(i - 1) % n] - points[i]
            v2 = points[(i + 1) % n] - points[i]
            v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
            v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
            cos_value = (v1 * v2).sum(axis=1)
            angles[:, i] = np.arccos(cos_value)

        return angles

    @functools.lru_cache(maxsize=1)
    def calculate_jacobians(self):
        """Calculate jacobians of each element.
        Calculated jacobians are returned and also stored in
        the fem_data.elemental_data dict with key = 'jacobians' .

        Returns:
            jacobians: numpy.ndarray
        """
        if self.elements.element_type in ['tri']:
            jacobians = self._calculate_element_areas_tri() * 2
        elif self.elements.element_type in ['quad']:
            jacobians = self._calculate_jacobians_quad()
        else:
            raise NotImplementedError

        self.elemental_data.update({
            'jacobian':
            FEMAttribute('Jacobian', self.elements.ids, jacobians)})
        return jacobians

    def _calculate_jacobians_quad(self):
        elements = self.elements.data
        normal_vector = self.calculate_element_normals()
        points = [
            self.collect_node_positions_by_ids(elements[:, i])
            for i in range(4)]

        # projection onto a plane
        def projection(p):
            inner_product = (p * normal_vector).sum(axis=1)
            return p - inner_product[:, None] * normal_vector

        points = [projection(p) for p in points]

        vector1 = (-points[0] + points[1] + points[2] - points[3]) / 4
        vector2 = (-points[0] - points[1] + points[2] + points[3]) / 4
        cross = np.cross(vector1, vector2)
        return np.linalg.norm(cross, axis=1)

    @functools.lru_cache(maxsize=1)
    def calculate_element_normals(self):
        """Calculate normal vectors of each shell elements. Please note that
        the calculated normal may not be aligned with neighbor elements. To
        make vector field smooth, use femio.extract_direction_feature() method.

        Args:
        Returns:
            normals: numpy.ndarray
        """
        if self.elements.element_type in ['tri']:
            crosses = self._calculate_tri_crosses()
            normals = crosses / np.linalg.norm(crosses, axis=1, keepdims=True)
        elif self.elements.element_type in ['quad']:
            normals = self._calculate_quad_normals()
        else:
            raise NotImplementedError

        self.elemental_data.update({
            'normal': FEMAttribute('Normal', self.elements.ids, normals)})
        return normals

    def extract_direction_feature(self, vectors, *, skip_normalization=False):
        """Extract direction feature, which take into account v ~ -v
        symmetricity. The resultant vectors will not have the same directions
        as originals', but the vector field will be smoother.

        Args:
            vectors: numpy.ndarray
                (n, 3) shaped input vectors.
            skip_normalization: bool, optional [False]
                If True, skip normalization of vector.
        """
        if len(vectors.shape) != 2:
            raise ValueError(
                f"Input vector should be 2-dim array but {vectors.shape}")
        if vectors.shape[-1] != 3:
            raise ValueError(
                "Input vector should be 3 dimensional vector "
                f"but {vectors.shape[-1]}")

        if not skip_normalization:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        xs = vectors[:, 0]
        ys = vectors[:, 1]
        zs = vectors[:, 2]
        phies = np.arctan2(ys, xs)
        thetas = np.arctan2(np.sqrt(xs**2 + ys**2), zs)
        cos_2theta = np.cos(2 * thetas)
        return np.stack([
            (1 - cos_2theta) * np.cos(2 * phies),
            (1 - cos_2theta) * np.sin(2 * phies),
            cos_2theta
        ], axis=1)

    @functools.lru_cache(maxsize=1)
    def _calculate_tri_crosses(self):
        elements = self.elements.data[:, :4]
        node0_points = self.collect_node_positions_by_ids(elements[:, 0])
        node1_points = self.collect_node_positions_by_ids(elements[:, 1])
        node2_points = self.collect_node_positions_by_ids(elements[:, 2])
        v10 = node1_points - node0_points
        v20 = node2_points - node0_points

        crosses = np.cross(v10, v20)
        return crosses

    @functools.lru_cache(maxsize=1)
    def _calculate_quad_normals(self):
        elements = self.elements.data[:, :4]
        node0_points = self.collect_node_positions_by_ids(elements[:, 0])
        node1_points = self.collect_node_positions_by_ids(elements[:, 1])
        node2_points = self.collect_node_positions_by_ids(elements[:, 2])
        node3_points = self.collect_node_positions_by_ids(elements[:, 3])

        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        crosses1 = np.cross(v10, v20)
        v32 = node3_points - node2_points
        v02 = node0_points - node2_points
        crosses2 = np.cross(v32, v02)
        vectors = crosses1 + crosses2
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    @functools.lru_cache(maxsize=1)
    def calculate_element_volumes(
            self, *, raise_negative_volume=True, return_abs_volume=False):
        """Calculate volume of each element assuming that the geometry of
        higher order elements is the same as that of order 1 elements.
        Calculated volumes are returned and also stored in
        the fem_data.elemental_data dict with key = 'volume' .

        Args:
            raise_negative_volume: bool, optional [True]
                If True, raise ValueError when negative volume exists.
            return_abs_volume: bool, optional [False]
                If True, return absolute volume instead of signed volume.
        Returns:
            volumes: numpy.ndarray
        """
        if self.elements.element_type in ['tet', 'tet2']:
            volumes = self._calculate_element_volumes_tet_like()
        elif self.elements.element_type in ['hex']:
            volumes = self._calculate_element_volumes_hex()
        else:
            raise NotImplementedError

        # Handle negative volumes according to the settings
        if raise_negative_volume and np.any(volumes < 0.):
            raise ValueError('Negative volume found.')
        if return_abs_volume:
            volumes = np.abs(volumes)

        self.elemental_data.update({
            'volume': FEMAttribute('Volume', self.elements.ids, volumes)})
        return volumes

    def _calculate_element_volumes_tet_like(self):
        """Calculate volume of each tet-2 like elements assuming that the
        geometry of higher order elements is the same as that of order 1
        elements.

        Args:
        Returns:
            volumes: numpy.ndarray
        """
        elements = self.elements.data[:, :4]
        node0_points = self.collect_node_positions_by_ids(elements[:, 0])
        node1_points = self.collect_node_positions_by_ids(elements[:, 1])
        node2_points = self.collect_node_positions_by_ids(elements[:, 2])
        node3_points = self.collect_node_positions_by_ids(elements[:, 3])
        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        v30 = node3_points - node0_points
        return np.linalg.det(np.stack([v10, v20, v30], axis=1))[:, None] / 6.

    def _calculate_element_volumes_hex(self):
        """Calculate volume of each hex elements.

        Parameters
        ----------

        Returns
        -------
            volumes: numpy.ndarray
        """
        elements = self.elements.data
        p0 = self.collect_node_positions_by_ids(elements[:, 0])
        p1 = self.collect_node_positions_by_ids(elements[:, 1])
        p2 = self.collect_node_positions_by_ids(elements[:, 2])
        p3 = self.collect_node_positions_by_ids(elements[:, 3])
        p4 = self.collect_node_positions_by_ids(elements[:, 4])
        p5 = self.collect_node_positions_by_ids(elements[:, 5])
        p6 = self.collect_node_positions_by_ids(elements[:, 6])
        p7 = self.collect_node_positions_by_ids(elements[:, 7])
        return 1. / 6. * (
            + np.linalg.det(np.stack([p1 - p4, p0 - p4, p3 - p4], axis=1))
            + np.linalg.det(np.stack([p2 - p6, p1 - p6, p3 - p6], axis=1))
            + np.linalg.det(np.stack([p1 - p6, p4 - p6, p3 - p6], axis=1))
            + np.linalg.det(np.stack([p7 - p3, p4 - p3, p6 - p3], axis=1))
            + np.linalg.det(np.stack([p1 - p5, p4 - p5, p6 - p5], axis=1))
        )[:, None]

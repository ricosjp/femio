import functools

from numba import njit
import numpy as np

from .fem_attribute import FEMAttribute
from . import functions


class GeometryProcessorMixin:

    def calculate_element_centroids(self, element_type=None, elements=None):
        """Calculate centroid of each element.

        Parameters
        ----------
        element_type: str, optional
            Element type of the element.
        elements: femio.FEMAttribute, optional
            If fed, compute centroids for the fed one.

        Returns
        -------
        centroid: numpy.ndarray[float]
            [n_elemnt, 3] shaped array of centroid coordinates.
        """
        if elements is None:
            element_type = self.elements.element_type
            elements = self.elements
        if element_type is None:
            raise ValueError('Feed element_type when elements is fed')

        if element_type == 'tri':
            centroid = self.convert_nodal2elemental(self.nodes.data)
        elif element_type == 'quad':
            centroid = self._calculate_element_centroids_quad(elements)
        elif element_type == 'polygon':
            centroid = self._calculate_element_centroids_polygon(elements)
        elif element_type == 'mix':
            centroid = np.zeros((len(self.elements), 3))
            for k, e in self.elements.items():
                partial_centroid = self.calculate_element_centroids(
                    elements=e, element_type=k)
                centroid[self.elements.types == k] = partial_centroid
        else:
            raise NotImplementedError(element_type)

        return centroid

    def _calculate_element_centroids_quad(self, elements):
        element_data = elements.data
        node0_points = self.collect_node_positions_by_ids(element_data[:, 0])
        node1_points = self.collect_node_positions_by_ids(element_data[:, 1])
        node2_points = self.collect_node_positions_by_ids(element_data[:, 2])
        node3_points = self.collect_node_positions_by_ids(element_data[:, 3])

        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        crosses1 = np.cross(v10, v20)
        v32 = node3_points - node2_points
        v02 = node0_points - node2_points
        crosses2 = np.cross(v32, v02)
        areas1 = np.linalg.norm(crosses1, axis=1, keepdims=True) / 2
        areas2 = np.linalg.norm(crosses2, axis=1, keepdims=True) / 2

        center1 = (node0_points + node1_points + node2_points) / 3
        center2 = (node0_points + node2_points + node3_points) / 3

        return (center1 * areas1 + center2 * areas2) / (areas1 + areas2)

    def _calculate_element_centroids_polygon(self, elements):
        centroid = np.stack([
            self._calculate_element_centroid_polygon(e)
            for e in elements.data], axis=0)
        return centroid

    def _calculate_element_centroid_polygon(self, element):
        triangle_elements = self._trianglate_polygon(element)
        node0_points = self.collect_node_positions_by_ids(
            triangle_elements[:, 0])
        node1_points = self.collect_node_positions_by_ids(
            triangle_elements[:, 1])
        node2_points = self.collect_node_positions_by_ids(
            triangle_elements[:, 2])
        centers = (node0_points + node1_points + node2_points) / 3

        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        crosses = np.cross(v10, v20)
        return np.einsum(
            'e,ep->p',
            np.linalg.norm(crosses, axis=1), centers) \
            / np.linalg.norm(np.sum(crosses, axis=0))

    def calculate_element_areas(
            self, *, mode="centroid", raise_negative_area=False,
            return_abs_area=True, elements=None, element_type=None,
            update=True):
        """Calculate areas of each element assuming that the geometry of
        higher order elements is the same as that of order 1 elements.
        Calculated areas are returned and also stored in
        the fem_data.elemental_data dict with key = 'area' .

        Parameters
        ----------
        mode: str, optional ["centroid"]
            If "gaussian", areas are calculated by gaussian integral.
            If "linear", areas are calculated under the assumption that
            all elements is linear, and it runs a bit faster.
            If "centroid", areas are calculated by decomposing it into
            triangles at centroid. Therefore, the result doesn't depend on
            the node labels.
        raise_negative_area: bool, optional [False]
            If True, raise ValueError when negative area exists.
        return_abs_area: bool, optional [True]
            If True, return absolute area instead of signed area.
        elements: femio.FEMAttribute
            If fed, compute volumes for the fed one.

        Returns
        -------
        areas: numpy.ndarray
        """
        if elements is None:
            if 'area' in self.elemental_data:
                return self._validate_metric(
                    self.elemental_data.get_attribute_data('area'),
                    raise_negative_metric=raise_negative_area,
                    return_abs_metric=return_abs_area)
            element_type = self.elements.element_type
            elements = self.elements
        else:
            if element_type is None:
                if elements.name == 'ELEMENT':
                    element_type = elements.element_type
                else:
                    element_type = elements.name

        if element_type in ['tri']:
            areas = self._calculate_element_areas_tri(elements)
        elif element_type in ['quad']:
            if mode == "linear":
                areas = self._calculate_element_areas_quad(elements)
            elif mode == "gaussian":
                areas = self._calculate_element_areas_quad_gaussian(elements)
            elif mode == "centroid":
                areas = self._calculate_element_areas_quad_centroid(elements)
            else:
                raise ValueError("Unknown mode")
        elif element_type in ['polygon']:
            if mode == "centroid":
                areas = self._calculate_element_areas_polygon(elements)
            else:
                areas = self._calculate_element_areas_polygon_centroid(
                    elements)
        elif element_type in ['mix']:
            areas = np.zeros((len(self.elements), 1))
            for k, e in self.elements.items():
                partial_areas = self.calculate_element_areas(
                    elements=e, element_type=k, update=False)
                areas[self.elements.types == k] = partial_areas
        else:
            raise NotImplementedError(element_type)

        areas = self._validate_metric(
            areas, raise_negative_metric=raise_negative_area,
            return_abs_metric=return_abs_area)

        if update:
            self.elemental_data.update_data(
                self.elements.ids, {'area': areas}, allow_overwrite=True)
        return areas

    def _calculate_element_areas_tri(self, elements):
        crosses = self._calculate_tri_crosses(elements)
        return np.linalg.norm(crosses, axis=1, keepdims=True) / 2.

    def _calculate_element_areas_quad(self, elements):
        element_data = elements.data
        node0_points = self.collect_node_positions_by_ids(element_data[:, 0])
        node1_points = self.collect_node_positions_by_ids(element_data[:, 1])
        node2_points = self.collect_node_positions_by_ids(element_data[:, 2])
        node3_points = self.collect_node_positions_by_ids(element_data[:, 3])

        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        crosses1 = np.cross(v10, v20)
        v32 = node3_points - node2_points
        v02 = node0_points - node2_points
        crosses2 = np.cross(v32, v02)
        areas1 = np.linalg.norm(crosses1, axis=1, keepdims=True) / 2
        areas2 = np.linalg.norm(crosses2, axis=1, keepdims=True) / 2
        return areas1 + areas2

    def _calculate_element_areas_quad_gaussian(self, elements):
        element_data = elements.data
        x0, y0, z0 = self.collect_node_positions_by_ids(element_data[:, 0]).T
        x1, y1, z1 = self.collect_node_positions_by_ids(element_data[:, 1]).T
        x2, y2, z2 = self.collect_node_positions_by_ids(element_data[:, 2]).T
        x3, y3, z3 = self.collect_node_positions_by_ids(element_data[:, 3]).T

        def J00(xi, eta):
            return (x1 - x0) * (1 - eta) + (x2 - x3) * (1 + eta)

        def J01(xi, eta):
            return (x3 - x0) * (1 - xi) + (x2 - x1) * (1 + xi)

        def J10(xi, eta):
            return (y1 - y0) * (1 - eta) + (y2 - y3) * (1 + eta)

        def J11(xi, eta):
            return (y3 - y0) * (1 - xi) + (y2 - y1) * (1 + xi)

        def J20(xi, eta):
            return (z1 - z0) * (1 - eta) + (z2 - z3) * (1 + eta)

        def J21(xi, eta):
            return (z3 - z0) * (1 - xi) + (z2 - z1) * (1 + xi)

        res = 0
        p = 0.5773502692
        for (xi, eta) in ((p, p), (-p, p), (p, -p), (-p, -p)):
            Jx = J10(xi, eta) * J21(xi, eta) - J20(xi, eta) * J11(xi, eta)
            Jy = J20(xi, eta) * J01(xi, eta) - J00(xi, eta) * J21(xi, eta)
            Jz = J00(xi, eta) * J11(xi, eta) - J10(xi, eta) * J01(xi, eta)
            res += (Jx * Jx + Jy * Jy + Jz * Jz)**.5
        res /= 16
        return res.reshape(-1, 1)

    def _calculate_element_areas_quad_centroid(self, elements):
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p = (p0 + p1 + p2 + p3) / 4
        v0, v1, v2, v3 = p0 - p, p1 - p, p2 - p, p3 - p
        cross_0 = np.cross(v0, v1)
        cross_1 = np.cross(v1, v2)
        cross_2 = np.cross(v2, v3)
        cross_3 = np.cross(v3, v0)
        normals = sum([cross_0, cross_1, cross_2, cross_3])
        return np.linalg.norm(normals, axis=1, keepdims=True) * 0.5

    def _calculate_element_areas_polygon(self, elements):
        areas = np.stack([
            self._calculate_element_area_polygon(e)
            for e in elements.data], axis=0)[..., None]
        return areas

    def _calculate_element_area_polygon(self, element):
        triangle_elements = self._trianglate_polygon(element)
        return np.linalg.norm(np.sum(
            self._calculate_tri_crosses(triangle_elements), axis=0)) * .5

    def _calculate_element_areas_polygon_centroid(self, elements):
        areas = np.stack([
            self._calculate_element_area_polygon_centroid(e)
            for e in elements.data], axis=0)[..., None]
        return areas

    def _calculate_element_area_polygon_centroid(self, element):
        points = self.collect_node_positions_by_ids(element)
        n = len(points)
        p = points.mean(axis=0)
        n = len(points)
        normal = np.zeros(3, np.float32)
        for i in range(n):
            v1 = points[i - 1] - p
            v2 = points[i] - p
            normal += np.cross(v1, v2)
        return np.linalg.norm(normal) * 0.5

    def _calculate_element_volumes_hex_gaussian(self, elements):
        element_data = elements.data
        x0, y0, z0 = self.collect_node_positions_by_ids(element_data[:, 0]).T
        x1, y1, z1 = self.collect_node_positions_by_ids(element_data[:, 1]).T
        x2, y2, z2 = self.collect_node_positions_by_ids(element_data[:, 2]).T
        x3, y3, z3 = self.collect_node_positions_by_ids(element_data[:, 3]).T
        x4, y4, z4 = self.collect_node_positions_by_ids(element_data[:, 4]).T
        x5, y5, z5 = self.collect_node_positions_by_ids(element_data[:, 5]).T
        x6, y6, z6 = self.collect_node_positions_by_ids(element_data[:, 6]).T
        x7, y7, z7 = self.collect_node_positions_by_ids(element_data[:, 7]).T

        def J0(xi, eta, zeta):
            c1 = (1 - eta) * (1 - zeta)
            c2 = (1 - eta) * (1 + zeta)
            c3 = (1 + eta) * (1 - zeta)
            c4 = (1 + eta) * (1 + zeta)
            J00 = c1 * (x1 - x0) + c2 * (x5 - x4) + \
                c3 * (x2 - x3) + c4 * (x6 - x7)
            J10 = c1 * (y1 - y0) + c2 * (y5 - y4) + \
                c3 * (y2 - y3) + c4 * (y6 - y7)
            J20 = c1 * (z1 - z0) + c2 * (z5 - z4) + \
                c3 * (z2 - z3) + c4 * (z6 - z7)
            return J00, J10, J20

        def J1(xi, eta, zeta):
            c1 = (1 - xi) * (1 - zeta)
            c2 = (1 - xi) * (1 + zeta)
            c3 = (1 + xi) * (1 - zeta)
            c4 = (1 + xi) * (1 + zeta)
            J01 = c1 * (x3 - x0) + c2 * (x7 - x4) + \
                c3 * (x2 - x1) + c4 * (x6 - x5)
            J11 = c1 * (y3 - y0) + c2 * (y7 - y4) + \
                c3 * (y2 - y1) + c4 * (y6 - y5)
            J21 = c1 * (z3 - z0) + c2 * (z7 - z4) + \
                c3 * (z2 - z1) + c4 * (z6 - z5)
            return J01, J11, J21

        def J2(xi, eta, zeta):
            c1 = (1 - xi) * (1 - eta)
            c2 = (1 - xi) * (1 + eta)
            c3 = (1 + xi) * (1 - eta)
            c4 = (1 + xi) * (1 + eta)
            J02 = c1 * (x4 - x0) + c2 * (x7 - x3) + \
                c3 * (x5 - x1) + c4 * (x6 - x2)
            J12 = c1 * (y4 - y0) + c2 * (y7 - y3) + \
                c3 * (y5 - y1) + c4 * (y6 - y2)
            J22 = c1 * (z4 - z0) + c2 * (z7 - z3) + \
                c3 * (z5 - z1) + c4 * (z6 - z2)
            return J02, J12, J22

        res = 0.0
        p = 0.5773502692
        for s in range(8):
            xi = p if s & 4 else -p
            eta = p if s & 2 else -p
            zeta = p if s & 1 else -p
            J00, J10, J20 = J0(xi, eta, zeta)
            J01, J11, J21 = J1(xi, eta, zeta)
            J02, J12, J22 = J2(xi, eta, zeta)
            res += J00 * J11 * J22 + J10 * J21 * J02 + J20 * J01 * J12
            res -= J00 * J21 * J12 + J10 * J01 * J22 + J20 * J11 * J02
        return res.reshape(-1, 1) / 512

    @functools.lru_cache(maxsize=1)
    def calculate_edge_lengths(self):
        """Calculate edge lengths of each element.
        Calculated lengths are returned and also stored in
        the fem_data.elemental_data dict with key = 'edge_lengths' .

        Returns:
            edge_lengths: numpy.ndarray
        """
        elements = self.elements
        if self.elements.element_type in ['tri', 'quad']:
            edge_lengths = self._calculate_edge_lengths_polygon(elements)
        else:
            raise NotImplementedError

        self.elemental_data.update_data(
            self.elements.ids, {'edge_lengths': edge_lengths},
            allow_overwrite=True)
        return edge_lengths

    def _calculate_edge_lengths_polygon(self, elements):
        n = elements.data.shape[1]
        points = [
            self.collect_node_positions_by_ids(elements.data[:, i])
            for i in range(n)]
        edge_lengths = np.empty(elements.data.shape, dtype=np.float64)

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
            angles = self._calculate_angles_polygon(self.elements)
        else:
            raise NotImplementedError

        self.elemental_data.update_data(
            self.elements.ids, {'angles': angles}, allow_overwrite=True)
        return angles

    def _calculate_angles_polygon(self, elements):
        n = elements.data.shape[1]
        points = [
            self.collect_node_positions_by_ids(elements.data[:, i])
            for i in range(n)]
        angles = np.empty(elements.data.shape, dtype=np.float64)

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
        elements = self.elements
        if self.elements.element_type in ['tri']:
            jacobians = self._calculate_element_areas_tri(elements) * 2
        elif self.elements.element_type in ['quad']:
            jacobians = self._calculate_jacobians_quad(elements)
        else:
            raise NotImplementedError

        self.elemental_data.update_data(
            self.elements.ids, {'jacobian': jacobians}, allow_overwrite=True)
        return jacobians

    def _calculate_jacobians_quad(self, elements):
        normal_vector = self.calculate_element_normals()
        points = [
            self.collect_node_positions_by_ids(elements.data[:, i])
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

    @functools.lru_cache(maxsize=2)
    def calculate_surface_normals(self, mode='mean'):
        """Calculate nodal normal vectors of the surface of a solid mesh.
        If an node is not on the surface, the vector will be zero.

        Args:
            mode: str, optional
                If 'mean', use mean metric to weight.
                If 'effective', use effective metric to weight.
                The default is 'mean'.
        Returns:
            normals: numpy.ndarray
                (n_node, 3)-shaped array.
        """
        surface_fem_data = self.to_surface()
        surface_normals = surface_fem_data.calculate_element_normals()
        surface_nodal_normals = functions.normalize(
            surface_fem_data.convert_elemental2nodal(
                surface_normals, mode=mode), keep_zeros=True)
        nodal_normals = FEMAttribute(
            'normal', self.nodes.ids, np.zeros((len(self.nodes.ids), 3)))
        nodal_normals.loc[surface_fem_data.nodes.ids].data = \
            surface_nodal_normals
        self.nodal_data.update({'normal': nodal_normals})
        return nodal_normals.data

    def calculate_normal_incidence_matrix(self):
        """Calculate incidence matrix based on the normal vector which point
        outside of each element.

        Returns
        -------
        facet_fem_data: femio.FEMData
        signed_incidence_matrix: scipy.sparse.csr_matrix[int]
            Positive if the direction of the facet's normal vector is in
            outside direction. Negative otherwise.
        facet_normal_vectors: numpy.ndarray[float]
            [n_facet, 3]-shaped array represents normal vectors of facets.
        """
        facet_data = self.to_facets(remove_duplicates=True)
        relative_incidence = self.calculate_relative_incidence_metrix_element(
            facet_data, minimum_n_sharing=None)  # TODO: Fix minimum_n_sharing
        coo = relative_incidence.tocoo()  # [n_cell, n_facet]

        cell_pos = self.convert_nodal2elemental(
            self.nodes.data, calc_average=True)
        facet_pos = facet_data.convert_nodal2elemental(
            facet_data.nodes.data, calc_average=True)
        facet_normals = facet_data.calculate_element_normals()
        rela_x = coo.multiply(facet_pos[:, 0]) - coo.T.multiply(
            cell_pos[:, 0]).T
        rela_y = coo.multiply(facet_pos[:, 1]) - coo.T.multiply(
            cell_pos[:, 1]).T
        rela_z = coo.multiply(facet_pos[:, 2]) - coo.T.multiply(
            cell_pos[:, 2]).T
        dots = rela_x.multiply(facet_normals[:, 0]) + rela_y.multiply(
            facet_normals[:, 1]) + rela_z.multiply(facet_normals[:, 2])
        dots.data[dots.data < 0] = -1
        dots.data[dots.data >= 0] = 1
        signed_incidence = coo.multiply(dots).tocsr()
        return facet_data, signed_incidence, facet_normals

    @functools.lru_cache(maxsize=1)
    def calculate_all_element_normals(self, facet_data=None):
        """Calculate normal vectors of each elements. If the elements are
        solid, then multiple normal vectors per solid will be generated.

        Returns
        -------
        normals: numpy.ndarray
            (n_element, n_faces_per_element, 3)-shaped array.
        facet_data: femio.FEMData
            FEMData object of facet data.
        """
        if self.elements.element_type in ['tet', 'tet2']:
            n_facet_per_element = 4
        elif self.elements.element_type == 'hex':
            n_facet_per_element = 6
        else:
            raise NotImplementedError(
                f"Unsupported element type: {self.elements.element_type}")

        if facet_data is None:
            facet_data = self.to_facets(remove_duplicates=False)
        normals = facet_data.calculate_element_normals()
        return np.reshape(
            normals, (len(self.elements), n_facet_per_element, 3))

    @functools.lru_cache(maxsize=1)
    def calculate_element_normals(
            self, mode="centroid", elements=None, element_type=None,
            update=True):
        """Calculate normal vectors of each shell elements. Please note that
        the calculated normal may not be aligned with neighbor elements. To
        make vector field smooth, use femio.extract_direction_feature() method.

        Args:
        mode: str, optional ["centroid"]
            If "centroid", normal vectors are calculated by decomposing
            it into triangles at centroid. Therefore, the result doesn't
            depend on the node labels.
        Returns:
            normals: numpy.ndarray
                (n_element, 3)-shaped array.
        """
        if elements is None:
            elements = self.elements
        if element_type is None:
            element_type = self.elements.element_type
        if element_type in ['tri']:
            normals = self._calculate_tri_normals(elements)
        elif element_type in ['quad']:
            if mode == "centroid":
                normals = self._calculate_quad_normals_centroid(elements)
            else:
                normals = self._calculate_quad_normals(elements)
        elif element_type in ['polygon']:
            if mode == "centroid":
                normals = self._calculate_polygon_normals_centroid(elements)
            else:
                normals = self._calculate_polygon_normals(elements)
        elif element_type in ['mix']:
            normals = np.zeros((len(self.elements), 3))
            for k, e in self.elements.items():
                partial_normals = self.calculate_element_normals(
                    elements=e, element_type=k, update=False)
                normals[self.elements.types == k] = partial_normals
        else:
            raise NotImplementedError(self.elements.element_type)
        normals = functions.normalize(normals)

        if update:
            self.elemental_data.update_data(
                self.elements.ids, {'normal': normals}, allow_overwrite=True)
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

    def _calculate_tri_normals(self, elements):
        crosses = self._calculate_tri_crosses(elements)
        normals = functions.normalize(crosses)
        return normals

    def _calculate_tri_crosses(self, elements):
        if isinstance(elements, np.ndarray):
            elements_data = elements
        else:
            elements_data = elements.data
        node0_points = self.collect_node_positions_by_ids(elements_data[:, 0])
        node1_points = self.collect_node_positions_by_ids(elements_data[:, 1])
        node2_points = self.collect_node_positions_by_ids(elements_data[:, 2])
        v10 = node1_points - node0_points
        v20 = node2_points - node0_points

        crosses = np.cross(v10, v20)
        return crosses

    def _calculate_quad_normals(self, elements):
        node0_points = self.collect_node_positions_by_ids(elements.data[:, 0])
        node1_points = self.collect_node_positions_by_ids(elements.data[:, 1])
        node2_points = self.collect_node_positions_by_ids(elements.data[:, 2])
        node3_points = self.collect_node_positions_by_ids(elements.data[:, 3])

        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        crosses1 = np.cross(v10, v20)
        v32 = node3_points - node2_points
        v02 = node0_points - node2_points
        crosses2 = np.cross(v32, v02)
        vectors = crosses1 + crosses2
        vectors = functions.normalize(vectors)
        return vectors

    def _calculate_quad_normals_centroid(self, elements):
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p = (p0 + p1 + p2 + p3) / 4
        v0, v1, v2, v3 = p0 - p, p1 - p, p2 - p, p3 - p
        cross_0 = np.cross(v0, v1)
        cross_1 = np.cross(v1, v2)
        cross_2 = np.cross(v2, v3)
        cross_3 = np.cross(v3, v0)
        normals = sum([cross_0, cross_1, cross_2, cross_3])
        return functions.normalize(normals)

    def _calculate_polygon_normals(self, elements):
        vectors = functions.normalize(np.stack([
            self._calculate_polygon_cross(e) for e in elements.data], axis=0))
        return vectors

    def _calculate_polygon_normals_centroid(self, elements):
        vectors = functions.normalize(np.stack([
            self._calculate_polygon_cross_centroid(e) for e in elements.data],
            axis=0))
        return vectors

    def _calculate_polygon_cross(self, element):
        triangle_elements = self._trianglate_polygon(element)
        return np.mean(self._calculate_tri_crosses(triangle_elements), axis=0)

    def _calculate_polygon_cross_centroid(self, element):
        points = self.collect_node_positions_by_ids(element)
        n = len(points)
        p = points.mean(axis=0)
        n = len(points)
        normal = np.zeros(3, np.float32)
        for i in range(n):
            v1 = points[i - 1] - p
            v2 = points[i] - p
            normal += np.cross(v1, v2)
        return normal

    def _trianglate_polygon(self, element):
        n_points = len(element)
        assert n_points > 2, f"Non facet element fed (given {n_points} points)"
        return np.stack([
            element[[0, i + 1, i + 2]] for i in range(n_points - 2)], axis=0)

    def calculate_element_metrics(
            self, *, raise_negative_metric=True, return_abs_metric=False,
            elements=None, element_type=None, update=True):
        """Calculate metric (area or volume depending on the mesh dimension)
        of each element assuming that the geometry of
        higher order elements is the same as that of order 1 elements.
        Calculated metrics are returned and also stored in
        the fem_data.elemental_data dict with key = 'metric'.

        Parameters
        ----------
        raise_negative_metric: bool, optional [True]
            If True, raise ValueError when negative metric exists.
        return_abs_metric: bool, optional [False]
            If True, return absolute volume instead of signed metric.
        elements: femio.FEMAttribute
            If fed, compute volumes for the fed one.

        Returns
        -------
        metrics: numpy.ndarray
        """
        if elements is None:
            if 'metric' in self.elemental_data:
                return self._validate_metric(
                    self.elemental_data.get_attribute_data('metric'),
                    raise_negative_metric=raise_negative_metric,
                    return_abs_metric=return_abs_metric)
            element_type = self.elements.element_type
            elements = self.elements
        else:
            if element_type is None:
                if elements.name == 'ELEMENT':
                    element_type = elements.element_type
                else:
                    element_type = elements.name
        if element_type in ['tri', 'tri2', 'quad', 'quad2', 'polygon']:
            metrics = self.calculate_element_areas(
                raise_negative_area=raise_negative_metric,
                return_abs_area=return_abs_metric, elements=elements,
                update=update)
        elif element_type in ['tet', 'tet2', 'hex',
                              'hexprism', 'prism', 'polyhedron']:
            metrics = self.calculate_element_volumes(
                raise_negative_volume=raise_negative_metric,
                return_abs_volume=return_abs_metric, elements=elements,
                update=update)
        elif element_type == 'mix':
            metrics = np.zeros((len(self.elements), 1))
            for k, e in self.elements.items():
                partial_metrics = self.calculate_element_metrics(
                    raise_negative_metric=raise_negative_metric,
                    return_abs_metric=return_abs_metric,
                    elements=e, element_type=k, update=False)
                metrics[self.elements.types == k] = partial_metrics
        else:
            raise NotImplementedError(
                f"Unsupported element type: {element_type}")
        metrics = self._validate_metric(
            metrics, raise_negative_metric=raise_negative_metric,
            return_abs_metric=return_abs_metric)

        if update:
            self.elemental_data.update_data(
                self.elements.ids, {'metric': metrics})
        return metrics

    def _validate_metric(
            self, metric, *, raise_negative_metric, return_abs_metric):
        if raise_negative_metric and np.any(metric < 0.):
            raise ValueError(
                f"Negative metric found: {metric[metric < 0]}")
        if return_abs_metric:
            metric = np.abs(metric)
        return metric

    def calculate_element_volumes(
            self, *, mode="centroid", raise_negative_volume=True,
            return_abs_volume=False, elements=None, element_type=None,
            faces=None, update=True):
        """Calculate volume of each element assuming that the geometry of
        higher order elements is the same as that of order 1 elements.
        Calculated volumes are returned and also stored in
        the fem_data.elemental_data dict with key = 'volume' .

        Parameters
        ----------
        mode: str, optional ["centroid"]
            If "gaussian", volumes are calculated by gaussian integral.
            If "linear", volumes are calculated under the assumption that
            all elements are linear, and it runs a bit faster.
            If "centroid", volumes are calculated by decomposing their surfaces
            into triangles at centroid. Therefore, the result doesn't depend on
            the node labels.
        raise_negative_volume: bool, optional [True]
            If True, raise ValueError when negative volume exists.
        return_abs_volume: bool, optional [False]
            If True, return absolute volume instead of signed volume.
        elements: femio.FEMAttribute
            If fed, compute volumes for the fed one.

        Returns
        -------
        volumes: numpy.ndarray
        """
        if elements is None:
            if 'volume' in self.elemental_data:
                return self._validate_metric(
                    self.elemental_data.get_attribute_data('volume'),
                    raise_negative_metric=raise_negative_volume,
                    return_abs_metric=return_abs_volume)
            element_type = self.elements.element_type
            elements = self.elements
            if element_type == 'polyhedron':
                faces = self.elemental_data['face']['polyhedron'].data
        else:
            if element_type is None:
                if elements.name == 'ELEMENT':
                    element_type = elements.element_type
                else:
                    element_type = elements.name

        if element_type in ['tet', 'tet2']:
            volumes = self._calculate_element_volumes_tet_like(
                elements)
        elif element_type in ['hex']:
            if mode == "linear":
                volumes = self._calculate_element_volumes_hex(
                    elements)
            elif mode == "gaussian":
                volumes = self._calculate_element_volumes_hex_gaussian(
                    elements)
            elif mode == "centroid":
                volumes = self._calculate_element_volumes_hex_centroid(
                    elements)
            else:
                raise ValueError("Unknown mode")
        elif element_type in ['pyr']:
            if mode == "centroid":
                volumes = self._calculate_element_volumes_pyr_centroid(
                    elements)
            else:
                volumes = self._calculate_element_volumes_pyr(elements)
        elif element_type in ['prism']:
            if mode == "centroid":
                volumes = self._calculate_element_volumes_prism_centroid(
                    elements)
            else:
                volumes = self._calculate_element_volumes_prism(elements)
        elif element_type in ['hexprism']:
            if mode == "centroid":
                volumes = self._calculate_element_volumes_hexprism(
                    elements)
            else:
                volumes = self._calculate_element_volumes_hexprism(
                    elements)
        elif element_type in ['polyhedron']:
            if mode == "centroid":
                volumes = self._calculate_element_volumes_polyhedron_centroid(
                    faces)
            else:
                volumes = self._calculate_element_volumes_polyhedron(faces)
        elif element_type == 'mix':
            volumes = np.zeros((len(self.elements), 1))
            for k, e in self.elements.items():
                if k == 'polyhedron':
                    faces = self.elemental_data['face']['polyhedron'].data
                else:
                    faces = None
                partial_volumes = self.calculate_element_volumes(
                    elements=e, element_type=k, update=False, mode=mode,
                    raise_negative_volume=raise_negative_volume,
                    return_abs_volume=return_abs_volume,
                    faces=faces
                )
                volumes[self.elements.types == k] = partial_volumes
        else:
            raise NotImplementedError(element_type, elements)

        volumes = self._validate_metric(
            volumes, raise_negative_metric=raise_negative_volume,
            return_abs_metric=return_abs_volume)

        if update:
            self.elemental_data.update_data(
                elements.ids, {'volume': volumes}, allow_overwrite=True)
        return volumes

    def _calculate_element_volumes_tet_like(self, elements):
        """Calculate volume of each tet-2 like elements assuming that the
        geometry of higher order elements is the same as that of order 1
        elements.

        Parameters
        ----------
        elements: femio.FEMAttribute
            Elements to calcluate volumes.

        Returns:
        volumes: numpy.ndarray
        """
        node0_points = self.collect_node_positions_by_ids(elements.data[:, 0])
        node1_points = self.collect_node_positions_by_ids(elements.data[:, 1])
        node2_points = self.collect_node_positions_by_ids(elements.data[:, 2])
        node3_points = self.collect_node_positions_by_ids(elements.data[:, 3])
        return self._calculate_element_volumes_tet_like_core(
            node0_points, node1_points, node2_points, node3_points)

    def _calculate_element_volumes_tet_like_core(
            self, node0_points, node1_points, node2_points, node3_points):
        v10 = node1_points - node0_points
        v20 = node2_points - node0_points
        v30 = node3_points - node0_points
        return np.linalg.det(np.stack([v10, v20, v30], axis=1))[:, None] / 6.

    def _calculate_element_volumes_hex(self, elements):
        """Calculate volume of each hex elements.

        Parameters
        ----------
        elements: femio.FEMAttribute
            Elements to calcluate volumes.

        Returns
        -------
        volumes: numpy.ndarray
        """
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        p5 = self.collect_node_positions_by_ids(elements.data[:, 5])
        p6 = self.collect_node_positions_by_ids(elements.data[:, 6])
        p7 = self.collect_node_positions_by_ids(elements.data[:, 7])
        return self._calculate_element_volumes_hex_with_nodes(
            p0, p1, p2, p3, p4, p5, p6, p7)

    def _calculate_element_volumes_hex_with_nodes(
            self, p0, p1, p2, p3, p4, p5, p6, p7):
        return 1. / 6. * (
            + np.linalg.det(np.stack([p1 - p4, p0 - p4, p3 - p4], axis=1))
            + np.linalg.det(np.stack([p2 - p6, p1 - p6, p3 - p6], axis=1))
            + np.linalg.det(np.stack([p1 - p6, p4 - p6, p3 - p6], axis=1))
            + np.linalg.det(np.stack([p7 - p3, p4 - p3, p6 - p3], axis=1))
            + np.linalg.det(np.stack([p1 - p5, p4 - p5, p6 - p5], axis=1))
        )[:, None]

    def _calculate_volumes_quad_centroid(self, p0, p1, p2, p3):
        n = len(p0)
        p = (p0 + p1 + p2 + p3) / 4
        res = np.zeros(n, np.float32)
        res += np.linalg.det(np.stack([p, p0, p1], axis=1))
        res += np.linalg.det(np.stack([p, p1, p2], axis=1))
        res += np.linalg.det(np.stack([p, p2, p3], axis=1))
        res += np.linalg.det(np.stack([p, p3, p0], axis=1))
        return res

    def _calculate_element_volumes_hex_centroid(self, elements):
        n = len(elements.data)
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        p5 = self.collect_node_positions_by_ids(elements.data[:, 5])
        p6 = self.collect_node_positions_by_ids(elements.data[:, 6])
        p7 = self.collect_node_positions_by_ids(elements.data[:, 7])
        res = np.zeros(n, np.float32)
        res += self._calculate_volumes_quad_centroid(p3, p2, p1, p0)
        res += self._calculate_volumes_quad_centroid(p5, p4, p0, p1)
        res += self._calculate_volumes_quad_centroid(p6, p7, p4, p5)
        res += self._calculate_volumes_quad_centroid(p2, p3, p7, p6)
        res += self._calculate_volumes_quad_centroid(p5, p1, p2, p6)
        res += self._calculate_volumes_quad_centroid(p4, p7, p3, p0)
        return (res / 6).reshape(n, 1)

    def _calculate_element_volumes_pyr(self, elements):
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        return self._calculate_element_volumes_tet_like_core(
            p0, p1, p2, p4) + self._calculate_element_volumes_tet_like_core(
                p0, p2, p3, p4)

    def _calculate_element_volumes_pyr_centroid(self, elements):
        n = len(elements.data)
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        vol = np.zeros(n, np.float32)
        vol += np.linalg.det(np.stack([p0, p1, p4], axis=1))
        vol += np.linalg.det(np.stack([p1, p2, p4], axis=1))
        vol += np.linalg.det(np.stack([p2, p3, p4], axis=1))
        vol += np.linalg.det(np.stack([p3, p0, p4], axis=1))
        vol += self._calculate_volumes_quad_centroid(p1, p0, p3, p2)
        return (vol / 6).reshape(n, -1)

    def _calculate_element_volumes_prism(self, elements):
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        p5 = self.collect_node_positions_by_ids(elements.data[:, 5])
        return self._calculate_element_volumes_tet_like_core(
            p0, p2, p1, p3) \
            + self._calculate_element_volumes_tet_like_core(
                p1, p3, p2, p4) \
            + self._calculate_element_volumes_tet_like_core(
                p2, p4, p3, p5)

    def _calculate_element_volumes_prism_centroid(self, elements):
        n = len(elements.data)
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        p5 = self.collect_node_positions_by_ids(elements.data[:, 5])
        vol = np.zeros(n, np.float32)
        vol += np.linalg.det(np.stack([p0, p1, p2], axis=1))
        vol += np.linalg.det(np.stack([p5, p4, p3], axis=1))
        vol += self._calculate_volumes_quad_centroid(p2, p5, p3, p0)
        vol += self._calculate_volumes_quad_centroid(p1, p4, p5, p2)
        vol += self._calculate_volumes_quad_centroid(p0, p3, p4, p1)
        return (vol / 6).reshape(n, -1)

    def _calculate_element_volumes_hexprism(self, elements):
        """Calculate volume of each hexprism elements.

        Parameters
        ----------
        elements: femio.FEMAttribute
            Elements to calcluate volumes.

        Returns
        -------
        volumes: numpy.ndarray
        """
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        p5 = self.collect_node_positions_by_ids(elements.data[:, 5])
        p6 = self.collect_node_positions_by_ids(elements.data[:, 6])
        p7 = self.collect_node_positions_by_ids(elements.data[:, 7])
        p8 = self.collect_node_positions_by_ids(elements.data[:, 8])
        p9 = self.collect_node_positions_by_ids(elements.data[:, 9])
        p10 = self.collect_node_positions_by_ids(elements.data[:, 10])
        p11 = self.collect_node_positions_by_ids(elements.data[:, 11])
        return self._calculate_element_volumes_hex_with_nodes(
            p0, p1, p2, p3, p6, p7, p8, p9) \
            + self._calculate_element_volumes_hex_with_nodes(
                p0, p3, p4, p5, p6, p9, p10, p11)

    @staticmethod
    @njit
    def _calculate_element_volumes_polyhedron_core(face_csr, nodes):
        indptr, dat = face_csr
        P = len(indptr) - 1
        volumes = np.empty(P)
        for p in range(P):
            poly = dat[indptr[p]:indptr[p + 1]]
            n = poly[0]
            L = 1
            volume = 0.0
            for _ in range(n):
                k = poly[L]
                L += 1
                F = nodes[poly[L:L + k]]
                L += k
                for i in range(2, k):
                    A = F[0]
                    B = F[i - 1]
                    C = F[i]
                    volume += np.dot(np.cross(A, B), C)
            volumes[p] = volume / 6
        return volumes

    def _calculate_element_volumes_polyhedron(self, faces):
        """Calculate volume of each polyhedron elements.

        Parameters
        ----------
        faces: np.ndarray (array of (list of int))
            Face data of polyhedrons to calculate volumes.
        Returns
        -------
        volumes: numpy.ndarray
        """
        indptr = np.append(0, np.array([len(x) for x in faces], np.int64))
        np.cumsum(indptr, out=indptr)
        face_csr = (indptr, np.concatenate(faces))
        nodes = self.nodes.data
        volumes = self._calculate_element_volumes_polyhedron_core(
            face_csr, nodes)
        return volumes.reshape((-1, 1))

    @staticmethod
    @njit
    def _calculate_element_volumes_polyhedron_centroid_core(face_csr, nodes):
        indptr, dat = face_csr
        P = len(indptr) - 1
        volumes = np.empty(P)
        for p in range(P):
            poly = dat[indptr[p]:indptr[p + 1]]
            n = poly[0]
            L = 1
            volume = 0.0
            for _ in range(n):
                k = poly[L]
                L += 1
                F = nodes[poly[L:L + k]]
                L += k
                centroid = np.zeros(3, np.float32)
                for i in range(k):
                    centroid += F[i]
                centroid /= k
                for i in range(k):
                    volume += np.dot(np.cross(centroid, F[i - 1]), F[i])
            volumes[p] = volume / 6
        return volumes

    def _calculate_element_volumes_polyhedron_centroid(self, faces):
        indptr = np.append(0, np.array([len(x) for x in faces], np.int64))
        np.cumsum(indptr, out=indptr)
        face_csr = (indptr, np.concatenate(faces))
        nodes = self.nodes.data
        volumes = self._calculate_element_volumes_polyhedron_centroid_core(
            face_csr, nodes)
        return volumes.reshape((-1, 1))

    def make_elements_positive(self):
        """Perfmute element connectivity order when it has negative volume."""
        metric = self.calculate_element_metrics(
            raise_negative_metric=False)[:, 0]
        cond = metric < 0
        if np.sum(cond) == 0:
            return

        elements = self.elements.data
        elements[cond] = self._permute(self.elements.data[cond])
        self.elements.data = elements
        return

    def _permute(self, elements):
        if self.elements.element_type == 'mix':
            raise NotImplementedError
        if len(elements) == 0:
            return elements

        if self.elements.element_type == 'tri':
            raise ValueError('Should not reach here')
        elif self.elements.element_type == 'tet':
            return np.stack([
                elements[:, 0], elements[:, 2],
                elements[:, 1], elements[:, 3]], axis=-1)
        else:
            raise NotImplementedError

    def translation(self, vx, vy, vz):
        """Translate the nodes.
        If nodal_data or elemental_data exists,
        NotImplementedError is raised.

        Parameters
        ----------
        vx, vy, vz: float
            Coordinates of the translation vector.
        """
        if self.nodal_data.data or self.elemental_data.data:
            raise NotImplementedError
        self.nodes.data[:, 0] += vx
        self.nodes.data[:, 1] += vy
        self.nodes.data[:, 2] += vz

    def rotation(self, vx, vy, vz, theta):
        """Rotate the nodes, around straight line which throw
        (0,0,0) and (vx,vy,vz).

        Parameters
        ----------
        vx, vy, vz: float
            The axis of the rotation.
        theta: float
            Rotation angle.
        """
        norm = (vx * vx + vy * vy + vz * vz)**.5
        n1, n2, n3 = vx / norm, vy / norm, vz / norm

        if self.nodal_data.data or self.elemental_data.data:
            raise NotImplementedError
        X = self.nodes.data[:, 0]
        Y = self.nodes.data[:, 1]
        Z = self.nodes.data[:, 2]
        c, s = np.cos(theta), np.sin(theta)
        coefs = [n1 * n1 * (1 - c) + c,
                 n1 * n2 * (1 - c) - n3 * s,
                 n1 * n3 * (1 - c) + n2 * s]
        new_X = coefs[0] * X + coefs[1] * Y + coefs[2] * Z
        coefs = [n2 * n2 * (1 - c) + c,
                 n2 * n3 * (1 - c) - n1 * s,
                 n2 * n1 * (1 - c) + n3 * s]
        new_Y = coefs[0] * Y + coefs[1] * Z + coefs[2] * X
        coefs = [n3 * n3 * (1 - c) + c,
                 n3 * n1 * (1 - c) - n2 * s,
                 n3 * n2 * (1 - c) + n1 * s]
        new_Z = coefs[0] * Z + coefs[1] * X + coefs[2] * Y
        self.nodes.data[:, 0] = new_X
        self.nodes.data[:, 1] = new_Y
        self.nodes.data[:, 2] = new_Z

    def integrate_node_attribute_over_surface(self, attr_name):
        """
        Integrate a node attribute over surface areas.

        Parameters
        ----------
        attr_name: str
            The name of node attribute.

        Returns
        -------
        integrated_value: float number
        """
        surface = self.extract_surface()[0]
        surf_xyz = self.nodes.data[surface]
        p0, p1, p2 = surf_xyz[:, 0], surf_xyz[:, 1], surf_xyz[:, 2]
        v01 = p1 - p0
        v02 = p2 - p0
        v = np.cross(v01, v02)
        areas = (v * v).sum(axis=1) ** .5 / 2

        values = self.nodal_data.data[attr_name].data.ravel()
        values = values[surface].mean(axis=1)

        return (values * areas).sum()

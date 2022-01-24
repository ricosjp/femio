import functools

import numpy as np
import scipy.sparse as sp

from .fem_attribute import FEMAttribute
from . import functions


class GeometryProcessorMixin:

    def calculate_element_areas(
            self, *, linear=False, raise_negative_area=False,
            return_abs_area=True, elements=None, element_type=None,
            update=True):
        """Calculate areas of each element assuming that the geometry of
        higher order elements is the same as that of order 1 elements.
        Calculated areas are returned and also stored in
        the fem_data.elemental_data dict with key = 'area' .

        Parameters
        ----------
        linear: bool, optional [False]
            If False, areas are calculated by gaussian integral.
            If True, areas are calculated under the assumption that
            all elements is linear, and it runs a bit faster.
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
            if linear:
                areas = self._calculate_element_areas_quad(elements)
            else:
                areas = self._calculate_element_areas_quad_gaussian(elements)
        elif element_type in ['polygon']:
            areas = self._calculate_element_areas_polygon(elements)
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

    def _calculate_element_areas_polygon(self, elements):
        areas = np.stack([
            self._calculate_element_area_polygon(e)
            for e in elements.data], axis=0)[..., None]
        return areas

    def _calculate_element_area_polygon(self, element):
        triangle_elements = self._trianglate_polygon(element)
        return np.linalg.norm(np.sum(
            self._calculate_tri_crosses(triangle_elements), axis=0)) * .5

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
        nodal_normals.loc[surface_fem_data.nodes.ids].data\
            = surface_nodal_normals
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
            facet_data, minimum_n_sharing=3)
        coo = relative_incidence.tocoo()

        tuple_facets = self.extract_facets(
            remove_duplicates=False, method=np.stack)[
                self.elements.element_type]
        tuple_all_facets = tuple(f[coo.row] for f in tuple_facets)
        all_normals = self.calculate_all_element_normals()[coo.row]
        col_facet_elements = facet_data.elements.data[coo.col]
        facet_normals = facet_data.calculate_element_normals()
        col_facet_normals = facet_normals[coo.col]

        if len(tuple_all_facets) > 1:
            raise NotImplementedError
        else:
            all_facets = tuple_all_facets[0]

        inner_prods = np.concatenate([
            np.dot(
                all_normal[np.all(np.isin(all_facet, facet_element), axis=1)],
                facet_normal)
            for all_facet, all_normal, facet_element, facet_normal
            in zip(
                all_facets, all_normals,
                col_facet_elements, col_facet_normals)])
        if np.sum(np.logical_and(
                -1 + 1e-3 < inner_prods, inner_prods < 1 - 1e-3)) > 0:
            raise ValueError(
                f"Normal vector computation failed: {inner_prods}")
        signed_incidence_data = np.zeros(len(inner_prods), dtype=int)
        signed_incidence_data[1. - 1e-3 < inner_prods] = 1
        signed_incidence_data[inner_prods < -1. + 1e-3] = -1
        signed_incidence = sp.csr_matrix((
            signed_incidence_data, (coo.row, coo.col)))
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
            self, elements=None, element_type=None, update=True):
        """Calculate normal vectors of each shell elements. Please note that
        the calculated normal may not be aligned with neighbor elements. To
        make vector field smooth, use femio.extract_direction_feature() method.

        Args:
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
            normals = self._calculate_quad_normals(elements)
        elif element_type in ['polygon']:
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
        normals = crosses / np.linalg.norm(crosses, axis=1, keepdims=True)
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
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def _calculate_polygon_normals(self, elements):
        vectors = np.stack([
            self._calculate_polygon_normal(e) for e in elements.data], axis=0)
        return vectors

    def _calculate_polygon_normal(self, element):
        triangle_elements = self._trianglate_polygon(element)
        return np.mean(self._calculate_tri_normals(triangle_elements), axis=0)

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
        elif element_type in ['tet', 'tet2', 'hex', 'hexprism', 'prism']:
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
            self, *, linear=False, raise_negative_volume=True,
            return_abs_volume=False, elements=None, element_type=None,
            update=True):
        """Calculate volume of each element assuming that the geometry of
        higher order elements is the same as that of order 1 elements.
        Calculated volumes are returned and also stored in
        the fem_data.elemental_data dict with key = 'volume' .

        Parameters
        ----------
        linear: bool, optional [False]
            If False, areas are calculated by gaussian integral.
            If True, areas are calculated under the assumption that
            all elements is linear, and it runs a bit faster.
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
            if linear:
                volumes = self._calculate_element_volumes_hex(
                    elements)
            else:
                volumes = self._calculate_element_volumes_hex_gaussian(
                    elements)
        elif element_type in ['pyr']:
            volumes = self._calculate_element_volumes_pyr(elements)
        elif element_type in ['prism']:
            volumes = self._calculate_element_volumes_prism(elements)
        elif element_type in ['hexprism']:
            volumes = self._calculate_element_volumes_hexprism(
                elements)
        elif element_type == 'mix':
            volumes = np.zeros((len(self.elements), 1))
            for k, e in self.elements.items():
                partial_volumes = self.calculate_element_volumes(
                    elements=e, element_type=k, update=False, linear=linear,
                    raise_negative_volume=raise_negative_volume,
                    return_abs_volume=return_abs_volume)
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

    def _calculate_element_volumes_pyr(self, elements):
        p0 = self.collect_node_positions_by_ids(elements.data[:, 0])
        p1 = self.collect_node_positions_by_ids(elements.data[:, 1])
        p2 = self.collect_node_positions_by_ids(elements.data[:, 2])
        p3 = self.collect_node_positions_by_ids(elements.data[:, 3])
        p4 = self.collect_node_positions_by_ids(elements.data[:, 4])
        return self._calculate_element_volumes_tet_like_core(
            p0, p1, p2, p4) + self._calculate_element_volumes_tet_like_core(
                p0, p2, p3, p4)

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

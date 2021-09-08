import datetime as dt
import functools

import heapq
import networkx as nx
from numba import njit
import numpy as np
import scipy.sparse as sp

from .fem_attribute import FEMAttribute
from . import functions


class GraphProcessorMixin:

    def separate(self):
        """Separate the FEMData object into parts in terms of connected
        subgraphs.

        Returns
        -------
        list_fem_data: List[femio.FEMData]
            Connected subgraphs of the FEMData object. The components are
            in the order of the smallest node ids.
        """
        original_graph = nx.from_scipy_sparse_matrix(
            self.calculate_adjacency_matrix_element())
        list_element_indices = list(nx.connected_components(original_graph))
        unsorted_list_fem_data = [
            self.extract_with_element_indices(list(element_indices))
            for element_indices in list_element_indices]
        return sorted(
            unsorted_list_fem_data, key=lambda fd: np.min(fd.nodes.ids))

    def convert_id_elements_to_index_elements(self, element_ids=None):
        if element_ids is None:
            return self.nodes.ids2indices(self.elements.data)
        else:
            return self.nodes.ids2indices(
                self.elements.filter_with_ids[element_ids].data)

    @functools.lru_cache(maxsize=1)
    def extract_surface(self, elements=None, element_type=None):
        """Extract surface from solid mesh.

        Returns
        -------
        surface_indices:
            indices of nodes (not IDs).
        surface_positions:
            Positions of each nodes on the surface.
        """
        dict_facets = self.extract_facets()
        dict_facet_shapes = {'tri': [], 'quad': []}
        for facet in dict_facets.values():
            for f in facet:
                n_node_per_element = f.shape[-1]
                if n_node_per_element == 3:
                    dict_facet_shapes['tri'].append(f)
                elif n_node_per_element == 4:
                    dict_facet_shapes['quad'].append(f)
                else:
                    raise ValueError(
                        f"Unsupported element shape: {n_node_per_element}")

        extracted_surface_info = {
            k: self._extract_surface(np.concatenate(v, axis=0))
            for k, v in dict_facet_shapes.items() if len(v) > 0}
        if len(extracted_surface_info) == 1:
            s = list(extracted_surface_info.values())[0]
            return s[0], s[1]
        else:
            return {k: v[0] for k, v in extracted_surface_info.items()}, \
                {k: v[1] for k, v in extracted_surface_info.items()}

    def _extract_surface(self, facets):
        sorted_facets = np.array([np.sort(f) for f in facets])
        unique_sorted_facets, unique_indices, unique_counts = np.unique(
            sorted_facets, return_index=True, return_counts=True, axis=0)
        surface_ids = facets[unique_indices[np.where(unique_counts == 1)]]
        surface_indices = np.array(
            [[self.dict_node_id2index[node_id] for node_id in facet]
             for facet in surface_ids])
        surface_positions = np.array(
            [[self.nodes.data[index] for index in facet]
             for facet in surface_indices])
        return surface_indices, surface_positions

    def extract_surface_fistr(self):
        """Extract surface from solid mesh.

        Returns
        -------
        surface_data: 2D array of int.
            row data correspond to (element_id, surface_id) of surface.
        """
        data = self.elements.data
        N = len(data)

        # node_0, node_1, node_2, elem_id, surf_id
        surfs = np.empty((4 * N, 5), np.int32)
        surfs[0 * N:1 * N, :3] = data[:, [0, 1, 2]]
        surfs[1 * N:2 * N, :3] = data[:, [0, 1, 3]]
        surfs[2 * N:3 * N, :3] = data[:, [1, 2, 3]]
        surfs[3 * N:4 * N, :3] = data[:, [2, 0, 3]]
        surfs[0 * N:1 * N, 3] = self.elements.ids
        surfs[1 * N:2 * N, 3] = self.elements.ids
        surfs[2 * N:3 * N, 3] = self.elements.ids
        surfs[3 * N:4 * N, 3] = self.elements.ids
        surfs[0 * N:1 * N, 4] = 1
        surfs[1 * N:2 * N, 4] = 2
        surfs[2 * N:3 * N, 4] = 3
        surfs[3 * N:4 * N, 4] = 4

        surfs[:, :3].sort(axis=1)
        ind = np.lexsort(
            (surfs[:, 4], surfs[:, 3], surfs[:, 2], surfs[:, 1], surfs[:, 0]))
        surfs = surfs[ind]

        # select surce
        unique = np.ones(4 * N, np.bool_)
        distinct = (surfs[:-1, 0] != surfs[1:, 0])
        distinct |= (surfs[:-1, 1] != surfs[1:, 1])
        distinct |= (surfs[:-1, 2] != surfs[1:, 2])
        unique[:-1] &= distinct
        unique[1:] &= distinct

        surfs = surfs[unique]
        return surfs[:, 3:]

    def extract_facets(
            self, elements=None, element_type=None, remove_duplicates=False,
            method=None):
        """Extract facets.

        Parameters
        ----------
        elements: femio.FEMAttribute, optional
            If fed, extract facets of the specified elements.
        elements: str, optional
            If not fed, infer element type from the number of nodes per
            element.
        method: callable
            A method to aggregate facet features. If not fed, numpy.concatenate
            is used.

        Returns
        -------
        facets: dict[tuple(numpy.ndarray)]
        """
        if elements is None:
            elements = self.elements
        if element_type is None:
            if hasattr(elements, 'element_type'):
                element_type = elements.element_type
            else:
                nodes_per_element = elements.data.shape[1]
                if nodes_per_element == 3:
                    element_type = 'tri'
                elif nodes_per_element == 4:
                    element_type = 'tet'
                elif nodes_per_element == 10:
                    element_type = 'tet2'
                elif nodes_per_element == 8:
                    element_type = 'hex'
                elif nodes_per_element == 12:
                    element_type = 'hexprism'
                else:
                    raise ValueError(
                        f"Unknown nodes_per_element: {nodes_per_element}")

        if hasattr(elements, 'element_type'):
            if elements.element_type == 'mix':
                return {
                    element_type:
                    self.extract_facets(
                        element, element_type=element_type,
                        remove_duplicates=remove_duplicates, method=method)[
                            element_type]
                    for element_type, element in self.elements.items()}
            else:
                elements = list(elements.values())[0]

        if element_type == 'tri' or element_type == 'quad':
            facets = (elements.data,)
        else:
            facets = self._generate_all_faces(
                elements, element_type, method=method)

        if remove_duplicates:
            facets = tuple(functions.remove_duplicates(f) for f in facets)

        return {element_type: facets}

    def _generate_all_faces(
            self, elements=None, element_type=None, method=None):
        if elements is None:
            elements = self.elements
        if element_type is None:
            if hasattr(elements, 'element_type'):
                element_type = elements.element_type
            else:
                nodes_per_element = elements.data.shape[1]
                if nodes_per_element == 3:
                    element_type = 'tri'
                elif nodes_per_element == 4:
                    element_type = 'tet'
                elif nodes_per_element == 10:
                    element_type = 'tet2'
                elif nodes_per_element == 8:
                    element_type = 'hex'
                else:
                    raise ValueError(
                        f"Unknown nodes_per_element: {nodes_per_element}")

        if hasattr(elements, 'element_type'):
            root_data = {
                element_type:
                self.extract_surface(element, element_type=element_type)
                for element_type, element in self.elements.items()}
            return {
                e: d[0] for e, d in root_data.items()}, {
                    e: d[1] for e, d in root_data.items()}

        if method is None:
            method = np.concatenate

        if isinstance(elements, np.ndarray):
            elements_data = elements
        else:
            elements_data = elements.data

        if element_type == 'tri' or element_type == 'quad':
            face_ids = elements.data
        elif element_type == 'tet':
            face_ids = method([
                np.stack([
                    [element[0], element[2], element[1]],
                    [element[0], element[1], element[3]],
                    [element[1], element[2], element[3]],
                    [element[0], element[3], element[2]],
                ]) for element in elements_data])
        elif element_type == 'tet2':
            tet1_elements = elements_data[:, :4]
            face_ids = self._generate_all_faces(
                tet1_elements, 'tet', method=method)
        elif element_type == 'hex':
            face_ids = method([[
                [e[0], e[1], e[5], e[4]],
                [e[0], e[3], e[2], e[1]],
                [e[1], e[2], e[6], e[5]],
                [e[2], e[3], e[7], e[6]],
                [e[3], e[0], e[4], e[7]],
                [e[4], e[5], e[6], e[7]]]
                for e in elements_data])
        elif element_type == 'pyr':
            face_ids = (
                method([
                    [
                        [e[0], e[1], e[4]],
                        [e[1], e[2], e[4]],
                        [e[2], e[3], e[4]],
                        [e[3], e[0], e[4]],
                    ]
                    for e in elements_data]),
                method([
                    [
                        [e[0], e[3], e[2], e[1]],
                    ]
                    for e in elements_data]))
        elif element_type == 'prism':
            face_ids = (
                method([
                    [
                        [e[0], e[2], e[1]],
                        [e[3], e[4], e[5]],
                    ]
                    for e in elements_data]),
                method([
                    [
                        [e[0], e[1], e[4], e[3]],
                        [e[1], e[2], e[5], e[4]],
                        [e[0], e[3], e[5], e[2]],
                    ]
                    for e in elements_data]))
        elif element_type == 'hexprism':
            face_ids = method([[
                [e[0], e[5], e[4], e[1]],
                [e[1], e[4], e[3], e[2]],
                [e[5], e[11], e[10], e[4]],
                [e[4], e[10], e[9], e[3]],
                [e[3], e[9], e[8], e[2]],
                [e[0], e[6], e[11], e[5]],
                [e[6], e[7], e[10], e[11]],
                [e[7], e[8], e[9], e[10]],
                [e[1], e[2], e[8], e[7]],
                [e[0], e[1], e[7], e[6]]]
                for e in elements_data])
        else:
            raise NotImplementedError

        if isinstance(face_ids, tuple):
            return face_ids
        else:
            return (face_ids,)

    @functools.lru_cache(maxsize=1)
    def filter_first_order_nodes(self):
        """Obtain filter to get first order nodes.

        Returns
        -------
        filter: np.array() of bool
        """
        if self.elements.is_first_order():
            return np.ones(len(self.nodes.ids), dtype=bool)

        first_order_ids = np.unique(
            np.concatenate(self.elements.to_first_order().data))
        return np.isin(self.nodes.ids, first_order_ids, assume_unique=True)

    def calculate_filter_by_ids(self, all_ids, extracting_ids):
        filter_ = np.zeros(len(all_ids), dtype=bool)
        if len(extracting_ids) == (
                extracting_ids[-1] - extracting_ids[0] + 1) \
                and extracting_ids[0] == all_ids[0]:
            # extracting_ids are continuous and only former part of all_ids
            filter_[:len(extracting_ids)] = True
        else:
            filter_ = np.array([
                np.any(i == extracting_ids) for i in all_ids])
        return filter_

    def calculate_relative_incidence_metrix_element(
            self, other_fem_data, minimum_n_sharing):
        """Calculate incidence matrix from other_fem_data to self based on
        elements, i.e., the resultant incidence matrix being of the shape
        (n_self_element, n_other_element).

        Parameters
        ----------
        other_fem_data: femio.FEMData
            Other FEMData object.
        minimum_n_sharing: int
            The minimum number of sharing node ids to define edge connectivity.

        Returns
        -------
        incidence_matrix: scipy.sparse.csr_matrix
            Incidence matrix in CSR expression.
        """
        self_incidence = self.calculate_incidence_matrix().T.astype(
            int)  # (n_self_element, n_node)
        other_incidence = other_fem_data.calculate_incidence_matrix().astype(
            int)  # (n_node, n_other_element)
        relative_incidence = self_incidence.dot(other_incidence) \
            >= minimum_n_sharing  # (n_self_element, n_other_elemnt)
        return relative_incidence

    @functools.lru_cache(maxsize=1)
    def calculate_element_degree(self):
        """Calculated degrees of the graph of elements.
        Edges of the graph are defined by node shearing.
        Calculated degrees are returned and also stored in
        the fem_data.elemental_data dict with key = 'degree' .

        Returns
        -------
        degrees: numpy.ndarray
        """
        adj = self.calculate_adjacency_matrix()
        degrees = adj.sum(axis=1) - 1
        self.elemental_data.update_data(
            self.elements.ids, {'degree': degrees}, allow_overwrite=True)
        return degrees

    @functools.lru_cache(maxsize=2)
    def calculate_adjacency_matrix(
            self, *, mode='elemental', order1_only=True):
        """Calculate graph adjacency matrix.

        Parameters
        ----------
        mode: str, optional (['elemental'], 'nodal')
            If 'elemental', generate (n_element, n_element) shaped
            adjacency matrix where edges are defined by node shearing.
            If 'nodal', generate (n_node, n_node) shaped adjacency matrix
            with edges are defined by element shearing.
        order1_only: bool, optional [True]
            If True, consider only order 1 nodes. Effective only when
            mode == 'nodal'.

        Returns
        -------
        adj: scipy.sparse.csr_matrix
            Adjacency matrix in CSR expression.
        """
        if mode == 'elemental':
            adj = self.calculate_adjacency_matrix_element()
        elif mode == 'nodal':
            adj = self.calculate_adjacency_matrix_node(order1_only=order1_only)
        else:
            raise ValueError(f"Unexpected mode: {mode}")
        return adj

    @functools.lru_cache(maxsize=1)
    def calculate_adjacency_matrix_element(self):
        """Calculate graph adjacency matrix regarding elements sharing the same
        node as connected.

        Returns
        -------
        adj: scipy.sparse.csr_matrix
            Adjacency matrix in CSR expression.
        """
        print('Calculating incidence matrix')
        print(dt.datetime.now())
        # raise ValueError(node_ids, element_data, )
        incidence_matrix = self.calculate_incidence_matrix(
            order1_only=True)
        return incidence_matrix.T.dot(incidence_matrix).tocsr()

    @functools.lru_cache(maxsize=1)
    def calculate_adjacency_matrix_node(self, order1_only=True):
        """Calculate graph adjacency matrix regarding nodes connected with
        edges. Edges are defined by element shearing.

        Parameters
        ----------
        order1_only: bool, optional [True]
            If True, consider only order 1 nodes.

        Returns
        -------
        adj: scipy.sparse.csr_matrix
            Adjacency matrix in CSR expression.
        """
        print('Calculating incidence matrix')
        print(dt.datetime.now())
        # raise ValueError(node_ids, element_data, )
        incidence_matrix = self.calculate_incidence_matrix(
            order1_only=order1_only)
        return incidence_matrix.dot(incidence_matrix.T)

    @functools.lru_cache(maxsize=1)
    def calculate_incidence_matrix(self, order1_only=True):
        """Calculate graph incidence matrix, which is
        (n_node, n_element)-shaped matrix with bool.

        Parameters
        ----------
        order1_only: bool, optional
            If True, generate incidence matrix based on only order-one nodes.

        Returns
        -------
        incidence_matrix: scipy.sparse.csr_matrix
            (n_node, n_element)-shaped sparse matrix.
        """
        if order1_only:
            filter_ = self.filter_first_order_nodes()
            nodes = FEMAttribute(
                'ORDER1_NODE', self.nodes.ids[filter_],
                self.nodes.data[filter_], generate_id2index=True)
            elements = self.elements.to_first_order()
        else:
            nodes = self.nodes
            elements = self.elements

        node_indices = nodes.ids2indices(elements.data)
        element_indices = np.concatenate([
            [i] * len(n) for i, n in enumerate(node_indices)])
        incidence_matrix = sp.csr_matrix(
            (
                [True] * len(element_indices),
                (np.concatenate(node_indices), element_indices)),
            shape=(len(nodes), len(elements)))
        return incidence_matrix

    @functools.lru_cache(maxsize=15)
    def calculate_n_hop_adj(
            self, mode='elemental', n_hop=1, include_self_loop=True,
            order1_only=True):
        if mode == 'elemental':
            adj = self.calculate_adjacency_matrix_element()

        elif mode == 'nodal':
            adj = self.calculate_adjacency_matrix_node(
                order1_only=order1_only)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        adj = sp.csr_matrix(adj, dtype=bool)

        return_adj = adj
        power_adj = adj
        for i in range(1, n_hop):
            print(f"Calculating {i+1} hop adjacency matrix")
            power_adj = power_adj.dot(adj)
            return_adj = return_adj + power_adj

        if not include_self_loop:
            return_adj = sp.csr_matrix(
                return_adj - sp.eye(*adj.shape, dtype=int))
        return return_adj

    @functools.lru_cache(maxsize=4)
    def calculate_e2v_matrix(self, mode='elemental', include_self_loop=False):
        """Calculate aggregation matrix from elemental data to vertex data (
        in terms of either elemental or nodal graph).

        Parameters
        ----------
        mode : str, optional
            Mode of graph, which is either 'elemental' or 'nodal'. Defaults to
            'elemental'.
        include_self_loop : bool, optional
            If True, include self loop to the operation. Defaults to False.

        Returns
        -------
        e2v_matrix: scipy.sparse.coo_matrix
            Aggregation matrix in COO format.
        """
        if mode == 'elemental':
            adj = self.calculate_adjacency_matrix_element()
        elif mode == 'nodal':
            adj = self.calculate_adjacency_matrix_node()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if not include_self_loop:
            adj = sp.coo_matrix(adj - sp.eye(*adj.shape))

        data = np.ones(len(adj.row))
        new_col = np.arange(len(adj.row))
        e2v_matrix = sp.coo_matrix(
            (data, (adj.row, new_col)), shape=(adj.shape[0], len(adj.row)))
        return e2v_matrix

    def convert_element_data_to_indices(self, elements):
        return {
            element_type:
            np.array([
                self.collect_node_indices_by_ids(e) for e in elements_.data])
            for element_type, elements_ in elements.items()}

    def collect_node_indices_by_ids(self, node_ids):
        return np.array([
            self.dict_node_id2index[id_] for id_ in node_ids])

    def collect_node_positions_by_ids(self, node_ids):
        return self.nodes.data[self.collect_node_indices_by_ids(node_ids)]

    def collect_element_indices_by_ids(self, node_ids):
        return np.array([
            self.dict_element_id2index[id_] for id_ in node_ids])

    def collect_element_data_by_ids(self, node_ids):
        return self.elements.data[
            self.collect_element_indices_by_ids(node_ids)]

    @staticmethod
    @njit
    def _calculate_euclidean_hop_graph_nodal(
            indptr_v, indices_v, indptr_e, indices_e, node_pos, max_dist):
        eps = 1e-8
        max_dist += eps
        V = len(indptr_v) - 1
        E = len(indptr_e) - 1
        visited = np.zeros(V + E, np.bool_)
        res = [0] * 0

        for v in range(V):
            x, y, z = node_pos[v]

            def is_nbd(w):
                x1, y1, z1 = node_pos[w]
                dx, dy, dz = x - x1, y - y1, z - z1
                return dx*dx + dy*dy + dz*dz <= max_dist**2

            que = [v]
            visited[v] = 1
            for frm in que:
                if frm < V:
                    TO = indices_v[indptr_v[frm]:indptr_v[frm + 1]]
                    for to in TO:
                        to += V
                        if visited[to]:
                            continue
                        visited[to] = 1
                        que.append(to)
                else:
                    TO = indices_e[indptr_e[frm - V]:indptr_e[frm - V + 1]]
                    for to in TO:
                        if visited[to]:
                            continue
                        if not is_nbd(to):
                            continue
                        visited[to] = 1
                        que.append(to)
                        res.append(v)
                        res.append(to)
            for w in que:
                visited[w] = 0
        return res

    @staticmethod
    @njit
    def _calculate_euclidean_hop_graph_elemental(
            indptr_v, indices_v, indptr_e, indices_e, node_pos, max_dist):
        eps = 1e-8
        max_dist += eps
        V = len(indptr_v) - 1
        E = len(indptr_e) - 1
        visited = np.zeros(V + E, np.bool_)
        res = [0] * 0

        for e in range(E):
            v_ids = indices_e[indptr_e[e]:indptr_e[e+1]]
            xyz = node_pos[v_ids]

            def is_nbd(v):
                x, y, z = node_pos[v]
                for x1, y1, z1 in xyz:
                    dx, dy, dz = x - x1, y - y1, z - z1
                    if dx*dx + dy*dy + dz*dz <= max_dist**2:
                        return True
                return False

            que = [V + e]
            visited[V + e] = 1
            for frm in que:
                if frm < V:
                    TO = indices_v[indptr_v[frm]:indptr_v[frm + 1]]
                    for to in TO:
                        to += V
                        if visited[to]:
                            continue
                        visited[to] = 1
                        que.append(to)
                        res.append(e)
                        res.append(to - V)
                else:
                    TO = indices_e[indptr_e[frm - V]:indptr_e[frm - V + 1]]
                    for to in TO:
                        if visited[to]:
                            continue
                        if not is_nbd(to):
                            continue
                        visited[to] = 1
                        que.append(to)
            for w in que:
                visited[w] = 0
        return res

    def calculate_euclidean_hop_graph(self, r, *, mode='elemental'):
        """
        Calculate the adjacency matrix of graph G defined as follows.

        If mode is 'nodal', G is a nodal graph and node v, v' is
        adjacent in G if there exists a sequence of nodes
        (v_0, v_1, ..., v_n) satisfying
            - v_0 = v, v_n = w
            - dist(v, v_i) < r, for all i
            - v_i and v_{i+1} shares some element.

        If mode is 'elemental', G is a elemental graph and element e, e' is
        adjacent in G if there exists a sequence of elements
        (e_0, e_1, ..., e_n) satisfying
            - e_0 = e, e_n = e'
            - dist(e, e_i) < r, for all i
            - e_i and e_{i+1} shares some node.

        In elemental case, the distance of elements is defined by Euclidean
        distance between its vertex sets.

        In both cases, self-loops are excluded.

        Parameters
        ----------
        r : float
            radius of the ball.
        mode: str, optional (['elemental'], 'nodal')
            If 'elemental', generate (n_element, n_element) shaped
            adjacenty martix of the euclidean hop graph.
            If 'nodal', generate (n_node, n_node) shaped
            adjacenty martix of the euclidean hop graph.

        Returns
        -------
        adj: scipy.sparse.csr_matrix
            Adjacency matrix in CSR expression.
        """

        incidence = self.calculate_incidence_matrix()
        n_node, n_elem = incidence.shape
        indptr_v = incidence.indptr
        indices_v = incidence.indices
        incidence = incidence.T.tocsr()
        indptr_e = incidence.indptr
        indices_e = incidence.indices
        pos = self.nodes.data.astype(np.float64)

        if mode == 'nodal':
            adj_list = self._calculate_euclidean_hop_graph_nodal(
                indptr_v, indices_v, indptr_e, indices_e, pos, r)
            row, col = np.array(adj_list).reshape(-1, 2).T

            adj = sp.csr_matrix(
                ([True] * (len(adj_list) // 2), (row, col)),
                dtype=bool,
                shape=(n_node, n_node)
            )
        elif mode == 'elemental':
            adj_list = self._calculate_euclidean_hop_graph_elemental(
                indptr_v, indices_v, indptr_e, indices_e, pos, r)
            row, col = np.array(adj_list).reshape(-1, 2).T

            adj = sp.csr_matrix(
                ([True] * (len(adj_list) // 2), (row, col)),
                dtype=bool,
                shape=(n_elem, n_elem)
            )
        else:
            raise ValueError(f"Unexpected mode: {mode}")

        return adj
    
    @staticmethod
    @njit
    def distance_triangle_and_point(triangle, point):
        """Compute the euclid distance between a triangle and a point.

        Parameters
        ----------
        triangle : np.ndarray
            2D array of shape (3,3) containing float data.
            i-th row is the xyz-coorinates of i-th vertex of the input triangle.

        point : np.ndarray
            1D array containing float data.
            The xyz-coordinates of the input point.

        Returns
        -------
        nearest_point : np.ndarray
            The nearest point in the triangle from the input point.
        dist : float
            Computed distance.
        """
        p0, p1, p2 = triangle[0], triangle[1], triangle[2]
        p = point
        n = np.cross(p1 - p0, p2 - p0)
        n /= (n * n).sum()**.5
        n0 = np.cross(n, p2 - p1)
        n1 = np.cross(n, p0 - p2)
        n2 = np.cross(n, p1 - p0)
        side0 = not (((n0 * (p0 - p1)).sum() > 0) ^ ((n0 * (p - p1)).sum() > 0))
        side1 = not (((n1 * (p1 - p2)).sum() > 0) ^ ((n1 * (p - p2)).sum() > 0))
        side2 = not (((n2 * (p2 - p0)).sum() > 0) ^ ((n2 * (p - p0)).sum() > 0))
        if side0 and side1 and side2:
            k = (n * (p - p0)).sum()
            q = p - k * n
            return q, ((p - q)**2).sum()**.5

        opt_point = p0
        min_dist = ((p - p0)**2).sum()**.5

        for _ in range(3):
            p0, p1, p2 = p1, p2, p0
            d = p1 - p0
            t = ((p - p0) * d).sum()
            t /= (d * d).sum()
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            q = p0 + t * d
            dist = ((p - q)**2).sum()**.5
            if min_dist > dist:
                min_dist = dist
                opt_point = q
        return opt_point, min_dist

    @staticmethod
    @njit
    def _build_octree(points):
        """
        Build octree from the input points.
        Octree is a data-structure used in nearest-neighbor-search algorithm.

        Parameters
        ----------
        points : np.ndarray
            2D array of shape (N,3) containing float data.
            i-th row is the xyz-coorinates of i-th point of the input point.
        """

        MAX_SIZE = len(points) * 10
        while True:
            MAX_SIZE *= 2
            n_node = 1
            child = np.zeros((MAX_SIZE, 8), np.int64)
            node_point, sz = np.empty((MAX_SIZE, 2), np.int64), 0
            node_xyzw = np.empty((MAX_SIZE, 4), np.float64)
            a, b = points.min(), points.max()
            c = (a + b) / 2
            node_xyzw[0] = (c, c, c, (b - a) / 2)

            for p in range(len(points)):
                node_point[sz], sz = (0, p), sz + 1

            node_id = -1
            while node_id < n_node:
                node_id += 1
                left_i = np.searchsorted(node_point[:sz, 0], node_id, 'left')
                right_i = np.searchsorted(node_point[:sz, 0], node_id, 'right')
                nx, ny, nz, nw = node_xyzw[node_id]
                if right_i - left_i == 1:
                    node_id += 1
                    continue
                for k in range(8):
                    # center of the region of the new node.
                    cx = nx - nw / 2 if k & 4 else nx + nw / 2
                    cy = ny - nw / 2 if k & 2 else ny + nw / 2
                    cz = nz - nw / 2 if k & 1 else nz + nw / 2
                    cw = nw / 2
                    for i in range(left_i, right_i):
                        if sz == MAX_SIZE:
                            break
                        p = node_point[i, 1]
                        if bool(k & 4) ^ (points[p, 0] < nx):
                            continue
                        if bool(k & 2) ^ (points[p, 1] < ny):
                            continue
                        if bool(k & 1) ^ (points[p, 2] < nz):
                            continue
                        if child[node_id, k] == 0:
                            child_id, n_node = n_node, n_node + 1
                            node_xyzw[child_id] = (cx, cy, cz, cw)
                            child[node_id, k] = child_id
                        node_point[sz], sz = (child[node_id, k], p), sz + 1
            if sz == MAX_SIZE:
                continue
            child = child[:n_node]
            node_xyzw = node_xyzw[:n_node]
            node_point = node_point[:sz]
            idx = np.searchsorted(node_point[:, 0], np.arange(n_node + 1))
            return points, child, node_xyzw, node_point, idx
    
    def octree(self):
        if not hasattr(self, "_octree"):
            nodes = self.nodes.data.astype(np.float64)
            self._octree = self._build_octree(nodes)
        return self._octree

    def octree_c(self):
        if not hasattr(self, "_octree_c"):
            elements = self.elements.data
            shape = elements.shape
            elements = self.collect_node_positions_by_ids(elements.ravel())
            elements = elements.reshape(shape + (3, ))
            self._octree_c = self._build_octree(elements.sum(axis=1) / 3)
        return self._octree_c


    @staticmethod
    @njit
    def _nns_from_one_node_to_nodes(
        octree, node, k, distance_upper_bound
    ):
        p = node
        points, child, node_xyzw, node_point, idx = octree
        node_id = 0

        def possible_dist_min(xyzw, p):
            nx, ny, nz, nw = xyzw
            x = min(nx + nw, max(nx - nw, p[0]))
            y = min(ny + nw, max(ny - nw, p[1]))
            z = min(nz + nw, max(nz - nw, p[2]))
            return ((p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2)**.5

        que = [(0.0, 0)]  # priority queue min dist possible, node_id)
        res_q = [(-np.inf, -1)] * k
        
        while que:
            d, node_id = heapq.heappop(que)
            if d > -res_q[0][0] or d > distance_upper_bound:
                continue
            has_child = False
            for j in range(8):
                child_id = child[node_id, j]
                if child_id == 0:
                    continue
                has_child = True
                d = possible_dist_min(node_xyzw[child_id], p)
                heapq.heappush(que, (d, child_id))
            if has_child:
                continue
            for point_id in node_point[idx[node_id]:idx[node_id + 1], 1]:
                d = ((points[point_id] - p)**2).sum()**.5
                if d > distance_upper_bound:
                    continue
                heapq.heappushpop(res_q, (-d, point_id))
        nbd_indices = np.empty(k, np.int64)
        vectors = np.empty((k, 3), np.float64)
        for i in range(k):
            j = heapq.heappop(res_q)[1]
            nbd_indices[i] = j
            if j == -1:
                vectors[i] = (np.inf, np.inf, np.inf)
            else:
                vectors[i] = points[j] - p
        return nbd_indices[::-1], vectors[::-1]

    @staticmethod
    @njit
    def _hausdorff_distance_from_one_node(
        octree, node, max_dist_found
    ):
        p = node
        points, child, node_xyzw, node_point, idx = octree
        node_id = 0

        def possible_dist_min(xyzw, p):
            nx, ny, nz, nw = xyzw
            x = min(nx + nw, max(nx - nw, p[0]))
            y = min(ny + nw, max(ny - nw, p[1]))
            z = min(nz + nw, max(nz - nw, p[2]))
            return ((p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2)**.5

        que = [(0.0, 0)]  # priority queue (min dist possible, node_id)
        min_dist = np.inf
        while que:
            d, node_id = heapq.heappop(que)
            if d > min_dist:
                continue
            has_child = False
            for j in range(8):
                child_id = child[node_id, j]
                if child_id == 0:
                    continue
                has_child = True
                d = possible_dist_min(node_xyzw[child_id], p)
                heapq.heappush(que, (d, child_id))
            if has_child:
                continue
            for point_id in node_point[idx[node_id]:idx[node_id + 1], 1]:
                d = ((points[point_id] - p)**2).sum()**.5
                if d > min_dist:
                    continue
                min_dist = d
                if min_dist < max_dist_found:
                    return max_dist_found
        return min_dist


    @staticmethod
    @njit
    def _nns_from_one_element_to_nodes(
        octree, triangle, k, distance_upper_bound
    ):
        def distance_triangle_and_point(triangle, point):
            p0, p1, p2 = triangle[0], triangle[1], triangle[2]
            p = point
            n = np.cross(p1 - p0, p2 - p0)
            n /= (n * n).sum()**.5
            n0 = np.cross(n, p2 - p1)
            n1 = np.cross(n, p0 - p2)
            n2 = np.cross(n, p1 - p0)
            side0 = not (((n0 * (p0 - p1)).sum() > 0) ^ ((n0 * (p - p1)).sum() > 0))
            side1 = not (((n1 * (p1 - p2)).sum() > 0) ^ ((n1 * (p - p2)).sum() > 0))
            side2 = not (((n2 * (p2 - p0)).sum() > 0) ^ ((n2 * (p - p0)).sum() > 0))
            if side0 and side1 and side2:
                k = (n * (p - p0)).sum()
                q = p - k * n
                return ((p - q)**2).sum()**.5

            min_dist = ((p - p0)**2).sum()**.5

            for _ in range(3):
                p0, p1, p2 = p1, p2, p0
                d = p1 - p0
                t = ((p - p0) * d).sum()
                t /= (d * d).sum()
                if t < 0:
                    t = 0
                elif t > 1:
                    t = 1
                q = p0 + t * d
                dist = ((p - q)**2).sum()**.5
                if min_dist > dist:
                    min_dist = dist
            return min_dist
        
        p0, p1, p2 = triangle[0], triangle[1], triangle[2]
        p = (p0 + p1 + p2) / 3

        def dist(p, q):
            return ((p - q)**2).sum() ** .5
        d_triangle = max(dist(p, p0), dist(p, p1), dist(p, p2))

        points, child, node_xyzw, node_point, idx = octree
        node_id = 0

        def possible_dist_min(xyzw, p):
            nx, ny, nz, nw = xyzw
            x = min(nx + nw, max(nx - nw, p[0]))
            y = min(ny + nw, max(ny - nw, p[1]))
            z = min(nz + nw, max(nz - nw, p[2]))
            return ((p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2)**.5

        que = [(-d_triangle, 0)]  # priority queue min dist possible, node_id)
        res_q = [(-np.inf, -1)] * k
        while que:
            d, node_id = heapq.heappop(que)
            if d > -res_q[0][0] or d > distance_upper_bound:
                continue
            has_child = False
            for j in range(8):
                child_id = child[node_id, j]
                if child_id == 0:
                    continue
                has_child = True
                d = possible_dist_min(node_xyzw[child_id], p) - d_triangle
                heapq.heappush(que, (d, child_id))
            if has_child:
                continue
            for point_id in node_point[idx[node_id]:idx[node_id + 1], 1]:
                d = distance_triangle_and_point(triangle, points[point_id])
                if d > distance_upper_bound:
                    continue
                heapq.heappushpop(res_q, (-d, point_id))
        nbd_indices = np.empty(k, np.int64)
        distances = np.empty(k, np.float64)
        for i in range(k):
            distances[i], nbd_indices[i] = heapq.heappop(res_q)
            distances[i] *= -1
        return nbd_indices[::-1], distances[::-1]


    def _nns_from_one_node_to_elements(
        self, octree_c, triangles, node, k, distance_upper_bound
    ):
        k2 = k + k + 10
        nbd_ids, _ = self._nns_from_one_node_to_nodes(
            octree_c, node, k2, distance_upper_bound)
        distances = np.empty(k2, np.float64)
        for i in range(k2):
            if nbd_ids[i] == -1:
                distances[i] = np.inf
            else:
                triangle = triangles[nbd_ids[i]]
                distances[i] = self.distance_triangle_and_point(triangle, node)[1]
        order = np.argsort(distances)[:k]
        return nbd_ids[order], distances[order]


    @staticmethod
    @njit
    def _hausdorff_distance_between_two_triangles(triangle1, triangle2):
        def distance_triangle_and_point(triangle, point):
            p0, p1, p2 = triangle[0], triangle[1], triangle[2]
            p = point
            n = np.cross(p1 - p0, p2 - p0)
            n /= (n * n).sum()**.5
            n0 = np.cross(n, p2 - p1)
            n1 = np.cross(n, p0 - p2)
            n2 = np.cross(n, p1 - p0)
            side0 = not (((n0 * (p0 - p1)).sum() > 0) ^ ((n0 * (p - p1)).sum() > 0))
            side1 = not (((n1 * (p1 - p2)).sum() > 0) ^ ((n1 * (p - p2)).sum() > 0))
            side2 = not (((n2 * (p2 - p0)).sum() > 0) ^ ((n2 * (p - p0)).sum() > 0))
            if side0 and side1 and side2:
                k = (n * (p - p0)).sum()
                q = p - k * n
                return q, ((p - q)**2).sum()**.5

            min_dist = ((p - p0)**2).sum()**.5

            for _ in range(3):
                p0, p1, p2 = p1, p2, p0
                d = p1 - p0
                t = ((p - p0) * d).sum()
                t /= (d * d).sum()
                if t < 0:
                    t = 0
                elif t > 1:
                    t = 1
                q = p0 + t * d
                dist = ((p - q)**2).sum()**.5
                if min_dist > dist:
                    min_dist = dist
            return min_dist
        
        p0, p1, p2 = triangle1[0], triangle1[1], triangle1[2]
        q0, q1, q2 = triangle2[0], triangle2[1], triangle2[2]
        x0 = distance_triangle_and_point(triangle2, p0)
        x1 = distance_triangle_and_point(triangle2, p1)
        x2 = distance_triangle_and_point(triangle2, p2)
        y0 = distance_triangle_and_point(triangle1, q0)
        y1 = distance_triangle_and_point(triangle1, q1)
        y2 = distance_triangle_and_point(triangle1, q2)
        return max(x0, x1, x2, y0, y1, y2)        


    def nearest_neighbor_search_from_nodes_to_nodes(
        self, k, distance_upper_bound=np.inf, target_fem_data=None
    ):
        """
        Compute 1-st to k-th nearest nodes in target_fem_data
        from each nodes in this fem_data.
        
        Parameters
        ----------
        k : int
            The number of nodes to search.

        distance_upper_bound : float
            Search only neighbors within this distance.
        
        target_fem_data : FEMData Object

        Returns
        -------
        nbd_indices : np.ndarray
            2D array of shape (N, k) containing int data where N is
            the number of nodes.
            If the number of points found is less than k,
            corresponding value are set to be -1.

        vectors : np.ndarray
            3D array of shape (N, k, 3) containing float data.
            The vector from input point to neighbors.
            If the number of points found is less than k,
            corresponding values are filled by np.inf.
        """
        if target_fem_data is None:
            target_fem_data = self
        octree = target_fem_data.octree()
        nodes = self.nodes.data
        N = len(nodes)
        nbd_indices = np.empty((N, k), np.int64)
        vectors = np.empty((N, k, 3), np.float64)
        for i in range(N):
            nbd_indices[i], vectors[i] = \
                self._nns_from_one_node_to_nodes(
                    octree, nodes[i], k, distance_upper_bound)
        return nbd_indices, vectors


    def nearest_neighbor_search_from_elements_to_nodes(
        self, k, distance_upper_bound=np.inf, target_fem_data=None
    ):
        """
        Compute 1-st to k-th nearest nodes in target_fem_data
        from each elements in this fem_data.
        
        Parameters
        ----------
        k : int
            The number of nodes to search.

        distance_upper_bound : float
            Search only neighbors within this distance.

        target_fem_data : FEMData Object

        Returns
        -------
        nbd_indices : np.ndarray
            2D array of shape (N, k) containing int data where N is
            the number of nodes.
            If the number of points found is less than k,
            corresponding value are set to be -1.

        dists : np.ndarray
            2D array of shape (N, k, 3) containing float data.
            Distances between the elements and neighborhood nodes in
            target_fem_data. 
            If the number of points found is less than k,
            corresponding values are filled by np.inf.
        """
        if target_fem_data is None:
            target_fem_data = self
        elements = self.elements.data
        shape = elements.shape
        elements = self.collect_node_positions_by_ids(elements.ravel())
        elements = elements.reshape(shape + (3, ))
        octree = target_fem_data.octree()
        N = len(elements)
        nbd_indices = np.empty((N, k), np.int64)
        dists = np.empty((N, k), np.float64)
        for i in range(N):
            nbd_indices[i], dists[i] = \
                self._nns_from_one_element_to_nodes(
                    octree, elements[i], k, distance_upper_bound)
        return nbd_indices, dists


    def nearest_neighbor_search_from_nodes_to_elements(
        self, k, distance_upper_bound=np.inf, target_fem_data=None, 
    ):
        """
        Compute 1-st to k-th nearest elements in target_fem_data
        from each nodes in this fem_data.
        
        Parameters
        ----------
        k : int
            The number of nodes to search.

        distance_upper_bound : float
            Search only neighbors within this distance.

        target_fem_data : FEMData Object

        Returns
        -------
        nbd_indices : np.ndarray
            2D array of shape (N, k) containing int data where N is
            the number of nodes.
            If the number of points found is less than k,
            corresponding value are set to be -1.

        dists : np.ndarray
            2D array of shape (N, k, 3) containing float data.
            Distances between the elements and neighborhood nodes in
            target_fem_data. 
            If the number of points found is less than k,
            corresponding values are filled by np.inf.
        """
        nodes = self.nodes.data
        if target_fem_data is None:
            target_fem_data = self
        elements = self.elements.data
        shape = elements.shape
        elements = self.collect_node_positions_by_ids(elements.ravel())
        elements = elements.reshape(shape + (3, ))

        octree_c = target_fem_data.octree_c()
        
        N = len(nodes)
        nbd_indices = np.empty((N, k), np.int64)
        distances = np.empty((N, k), np.float64)

        for i in range(N):
            nbd_indices[i], distances[i] = \
                self._nns_from_one_node_to_elements(
                    octree_c, elements, nodes[i],
                    k, distance_upper_bound)
        return nbd_indices, distances
    

    def calculate_hausdorff_distance_nodes(
            self, target_fem_data, directed=False):
        """
        Calculate Hausdorff distance from this fem_data to
        target_fem_data.
        
        Directed Hausdorff distance from X to Y is difined by:
            HD(X to Y) := sup_{x} inf_{y} |x-y|.
        
        (Bidirected) Hausdorff distance is defined by
            min(HD(X to Y), HD(Y to X)). 
        
        Here, X and Y are point could (i.e. only nodes are considered).
        
        Parameters
        ----------
        target_fem_data : FEMData Object.

        directed : bool
            If True, calculate directed Hausdorff distance. 

        Returns
        -------
        dist : float
            Calculated Hausdorff distance. 
        """
        if directed:
            dist = -np.inf
            octree = target_fem_data.octree()
            for node in self.nodes.data:
                x = self._hausdorff_distance_from_one_node(
                    octree, node, dist)
                dist = max(dist, x)
            return dist
        else:
            HD1 = self.calculate_hausdorff_distance_nodes(
                target_fem_data, True)
            HD2 = target_fem_data.calculate_hausdorff_distance_nodes(
                self, True)
            return max(HD1, HD2)

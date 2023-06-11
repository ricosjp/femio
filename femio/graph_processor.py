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
        dict_facet_shapes = {'tri': [], 'quad': [], 'polygon': []}
        for facet in dict_facets.values():
            for f in facet:
                n_node_per_element = f.shape[-1]
                if n_node_per_element == 3:
                    dict_facet_shapes['tri'].append(f)
                elif n_node_per_element == 4:
                    dict_facet_shapes['quad'].append(f)
                else:
                    n_nodes = np.array([len(f_) for f_ in f])
                    unique_n_nodes = np.unique(n_nodes)
                    if 3 in unique_n_nodes:
                        dict_facet_shapes['tri'].append(
                            np.stack(f[n_nodes == 3]))
                    if 4 in unique_n_nodes:
                        dict_facet_shapes['quad'].append(
                            np.stack(f[n_nodes == 4]))
                    dict_facet_shapes['polygon'] += list(f[n_nodes > 4])

        extracted_surface_info = {
            k:
            self._extract_surface(np.array(v), facet_type=k)
            if k == 'polygon'
            else
            self._extract_surface(np.concatenate(v, axis=0), facet_type=k)
            for k, v in dict_facet_shapes.items() if len(v) > 0}
        if len(extracted_surface_info) == 1:
            s = list(extracted_surface_info.values())[0]
            return s[0], s[1]
        else:
            return {k: v[0] for k, v in extracted_surface_info.items()}, \
                {k: v[1] for k, v in extracted_surface_info.items()}

    def _extract_surface(self, facets, facet_type):
        if facet_type == 'polygon':
            sorted_facets = np.array(
                [np.sort(f) for f in facets], dtype=object)
            surface_indices, surface_positions \
                = self._extract_surface_polygon(facets, sorted_facets)
        else:
            sorted_facets = np.array([np.sort(f) for f in facets])
            unique_sorted_facets, unique_indices, unique_counts = np.unique(
                sorted_facets, return_index=True, return_counts=True, axis=0)
            surface_ids = facets[unique_indices[np.where(unique_counts == 1)]]
            surface_indices = self.nodes.ids2indices(surface_ids)
            surface_positions = self.nodes.data[surface_indices]
        return surface_indices, surface_positions

    def _extract_surface_polygon(self, facets, sorted_facets):
        n_nodes = np.array([len(f) for f in sorted_facets])
        unique_n_nodes = np.unique(n_nodes)
        list_surface_indices = []
        list_surface_positions = []
        n_surface = 0
        for n_node in unique_n_nodes:
            focus_sorted_facets = sorted_facets[n_nodes == n_node]
            unique_sorted_facets, unique_indices, unique_counts = np.unique(
                np.stack(focus_sorted_facets).astype(int),
                return_index=True, return_counts=True, axis=0)

            focus_facets = facets[n_nodes == n_node]
            surface_ids = focus_facets[
                unique_indices[np.where(unique_counts == 1)]]
            surface_indices = self.nodes.ids2indices(surface_ids)
            surface_positions = np.array([
                self.nodes.data[si] for si in surface_indices], dtype=object)
            n_surface += len(surface_ids)
            list_surface_indices.append(surface_indices)
            list_surface_positions.append(surface_positions)

        ret_surface_indices = np.empty(n_surface, object)
        ret_surface_indices[:] = [
            f_ for f in list_surface_indices for f_ in f]
        ret_surface_positions = np.empty(n_surface, object)
        ret_surface_positions[:] = [
            f_ for f in list_surface_positions for f_ in f]
        return ret_surface_indices, ret_surface_positions

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
                dict_element_facet = {
                    element_type:
                    self.extract_facets(
                        element, element_type=element_type,
                        remove_duplicates=False, method=method)[
                            element_type]
                    for element_type, element in self.elements.items()}
                n_vertices = np.unique([
                    len(f[0]) for facet in dict_element_facet.values()
                    for f in facet])

                dict_vertices_facet = {}
                for i_vertices in n_vertices:
                    dict_vertices_facet[i_vertices] = []
                    for key, facet in dict_element_facet.items():
                        for f in facet:
                            if len(f[0]) == i_vertices:
                                dict_vertices_facet[i_vertices].append(f)

                    if remove_duplicates:
                        dict_vertices_facet[i_vertices] \
                            = functions.remove_duplicates(np.concatenate(
                                dict_vertices_facet[i_vertices]))
                    else:
                        dict_vertices_facet[i_vertices] \
                            = np.concatenate(
                                dict_vertices_facet[i_vertices])

                return {'mix': tuple(v for v in dict_vertices_facet.values())}

            else:
                elements = list(elements.values())[0]

        if element_type == 'tri' or element_type == 'quad':
            facets = (elements.data,)
        else:
            facets = self._generate_all_faces(
                elements, element_type, method=method)

        if remove_duplicates:
            facets = tuple(
                functions.remove_duplicates(f) for f in facets)

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

        if element_type in ['tri', 'quad', 'polygon']:
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
                        [e[0], e[1], e[2]],
                        [e[3], e[5], e[4]],
                    ]
                    for e in elements_data]),
                method([
                    [
                        [e[0], e[3], e[4], e[1]],
                        [e[1], e[4], e[5], e[2]],
                        [e[0], e[2], e[5], e[3]],
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
        elif element_type == 'polyhedron':
            assert 'face' in self.elemental_data, \
                'No face definition found for polyhedron: ' \
                f"{self.elemental_data.keys()}"
            faces = [
                self._parse_polyhedron_faces(f)
                for f in self.elemental_data.get_attribute_data('face')]
            n_vertices = np.unique([
                len(_f) for f in faces for _f in f])

            # NOTE: It uses concatenate ignoring `method`
            if method == np.concatenate:
                face_ids = tuple(
                    np.concatenate(self._collect_faces_concat(faces, n))
                    for n in n_vertices)
            elif method == np.stack:
                face_ids = tuple(
                    np.array(self._collect_faces_stack(faces, n))
                    for n in n_vertices)
            else:
                raise ValueError(f"Unexpected method: {method}")
        else:
            raise NotImplementedError(
                f"Unexpected element type: {element_type}")

        if isinstance(face_ids, tuple):
            return face_ids
        else:
            return (face_ids,)

    def _collect_faces_concat(self, faces, n_vertex):
        collected_faces = [
            [f for f in face if len(f) == n_vertex]
            for face in faces
        ]
        return [np.stack(f) for f in collected_faces if len(f) > 0]

    def _collect_faces_stack(self, faces, n_vertex):
        return np.array([
            np.array([f for f in face if len(f) == n_vertex])
            for face in faces
        ])

    def _parse_polyhedron_faces(self, faces):
        def split(n, f):
            return f[:n], f[n:]

        d = faces[1:]
        parsed_faces = []
        for i in range(faces[0]):
            face, d = split(d[0], d[1:])
            parsed_faces.append(np.array(face) + 1)
        ret_faces = np.empty(len(parsed_faces), object)
        ret_faces[:] = parsed_faces
        return ret_faces

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
            self, other_fem_data, minimum_n_sharing=None):
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
        if minimum_n_sharing is None:
            n_vertex = np.array(other_incidence.sum(axis=0))[0]
            dot = self_incidence.dot(other_incidence).tocoo()
            col = dot.col
            filter_ = dot.data >= n_vertex[col]
            relative_incidence = sp.csr_matrix((
                np.ones(np.sum(filter_), dtype=bool),
                (dot.row[filter_], dot.col[filter_])), shape=dot.shape)
        else:
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

        node_indices = nodes.ids2indices(elements)
        if isinstance(node_indices, list):

            dict_element_indices = {
                k: elements.id2index.loc[v.ids].values[:, 0]
                for k, v in elements.items()}
            element_indices = np.concatenate([
                [eind] * len(n)
                for element_indices, ni
                in zip(dict_element_indices.values(), node_indices)
                for eind, n in zip(element_indices, ni)])
            flattened_node_indices = np.concatenate([
                np.concatenate(ni) for ni in node_indices])
        else:
            element_indices = np.concatenate([
                [i] * len(n) for i, n in enumerate(node_indices)])
            flattened_node_indices = np.concatenate(node_indices)

        incidence_matrix = sp.csr_matrix(
            (
                [True] * len(element_indices),
                (flattened_node_indices, element_indices)),
            shape=(len(nodes), len(elements)))
        return incidence_matrix

    @functools.lru_cache(maxsize=2)
    def calculate_laplacian_matrix(self, mode='nodal', order1_only=True):
        """Calculate edge-based graph incidence matrix, which is
        (n_edge, n_node)-shaped matrix with bool.

        Parameters
        ----------
        mode: str, optional, ['nodal', 'elemental']
        order1_only: bool, optional
            If True, generate incidence matrix based on only order-one nodes.

        Returns
        -------
        incidence_matrix: scipy.sparse.csr_matrix
            (n_edge, n_node)-shaped sparse matrix.
        """
        if mode == 'nodal':
            adj = self.calculate_adjacency_matrix_node(
                order1_only=order1_only)
        elif mode == 'elemental':
            adj = self.calculate_adjacency_matrix_element(
                order1_only=order1_only)
        else:
            raise ValueError(f"Unexpected mode: {mode}")
        adj_wo_loop = adj.astype(int) - sp.eye(*adj.shape, dtype=int)
        degree = sp.diags(np.ravel(adj_wo_loop.sum(axis=1)), dtype=int)
        return adj_wo_loop - degree

    @functools.lru_cache(maxsize=1)
    def calculate_edge_gradient_matrix(self, mode='nodal', order1_only=True):
        """Calculate edge-based graph gradient matrix, which is
        (n_edge, n_vertex)-shaped matrix with bool. n_vertex can be either
        n_node or n_element, depending on the `mode` option.

        Parameters
        ----------
        mode: str, optional, ['nodal', 'elemental']
        order1_only: bool, optional
            If True, generate incidence matrix based on only order-one nodes.

        Returns
        -------
        gradient_matrix: scipy.sparse.csr_matrix
            (n_edge, n_node)-shaped sparse matrix.
        """
        if mode == 'nodal':
            adj = self.calculate_adjacency_matrix_node(
                order1_only=order1_only).tocoo()
        elif mode == 'elemental':
            adj = self.calculate_adjacency_matrix_element(
                order1_only=order1_only).tocoo()
        else:
            raise ValueError(f"Unexpected mode: {mode}")
        col = np.concatenate([
            np.array([r, c]) for r, c in zip(adj.row, adj.col) if c > r])
        row = np.arange(len(col)) // 2
        data = np.ones(len(col))
        data[1::2] = -1
        gradient_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(int(len(col) / 2), adj.shape[0]),
            dtype=int)
        return gradient_matrix

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
                self.collect_node_indices_by_ids(e) for e in elements_.data],
                dtype=elements_.data.dtype)
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
                return dx * dx + dy * dy + dz * dz <= max_dist**2

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
            v_ids = indices_e[indptr_e[e]:indptr_e[e + 1]]
            xyz = node_pos[v_ids]

            def is_nbd(v):
                x, y, z = node_pos[v]
                for x1, y1, z1 in xyz:
                    dx, dy, dz = x - x1, y - y1, z - z1
                    if dx * dx + dy * dy + dz * dz <= max_dist**2:
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
    def build_octree_node(points, boundingbox):
        N = len(points)
        xmin, xmax, ymin, ymax, zmin, zmax = boundingbox
        w0 = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.51
        maxsize = 19173961
        node_xyzw = np.empty((maxsize, 4))
        node_xyzw[0] = (
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            (zmin + zmax) / 2,
            w0)
        for v in range(1, maxsize):
            p = (v - 1) >> 3
            r = (v - 1) & 7
            px, py, pz, pw = node_xyzw[p]
            vw = pw / 2
            vx = px - vw if r & 4 else px + vw
            vy = py - vw if r & 2 else py + vw
            vz = pz - vw if r & 1 else pz + vw
            node_xyzw[v] = (vx, vy, vz, vw)

        node_pt, sz = np.empty((9 * N, 2), np.int32), 0

        def add(v, i):
            nonlocal sz
            node_pt[sz], sz = (v, i), sz + 1

        for i in range(N):
            x, y, z = points[i]
            v = 0
            add(v, i)
            for _ in range(8):
                vx, vy, vz, vw = node_xyzw[v]
                assert vx - vw <= x <= vx + vw
                assert vy - vw <= y <= vy + vw
                assert vz - vw <= z <= vz + vw
                for r in range(8):
                    c = (v << 3) + r + 1
                    cx, cy, cz, cw = node_xyzw[c]
                    if not (cx - cw <= x <= cx + cw):
                        continue
                    if not (cy - cw <= y <= cy + cw):
                        continue
                    if not (cz - cw <= z <= cz + cw):
                        continue
                    v = c
                    break
                add(v, i)
        ID = np.argsort(node_pt[:, 0])
        node_pt = node_pt[ID]
        idx = np.searchsorted(node_pt[:, 0], np.arange(maxsize + 1))
        return (points, node_xyzw, node_pt, idx)

    @staticmethod
    @njit
    def build_octree_element(triangles, boundingbox):
        N = len(triangles)
        n_sample = min(100 * N, 1 << 24)
        areas = np.empty(N)
        for n in range(N):
            A, B, C = triangles[n]
            x1, y1, z1 = B - A
            x2, y2, z2 = C - A
            x3 = y1 * z2 - y2 * z1
            y3 = z1 * x2 - z2 * x1
            z3 = x1 * y2 - x2 * y1
            areas[n] = (x3 * x3 + y3 * y3 + z3 * z3)**.5
        sample_sz = (np.append(0, np.cumsum(areas)) * n_sample //
                     areas.sum()).astype(np.int32)
        sample_sz = sample_sz[1:] - sample_sz[:-1]
        sample_sz = np.maximum(sample_sz, 1)
        n_sample = sample_sz.sum()

        xmin, xmax, ymin, ymax, zmin, zmax = boundingbox
        w0 = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.51
        maxsize = 19173961
        node_xyzw = np.empty((maxsize, 4))
        node_xyzw[0] = (
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            (zmin + zmax) / 2,
            w0)
        for v in range(1, maxsize):
            p = (v - 1) >> 3
            r = (v - 1) & 7
            px, py, pz, pw = node_xyzw[p]
            vw = pw / 2
            vx = px - vw if r & 4 else px + vw
            vy = py - vw if r & 2 else py + vw
            vz = pz - vw if r & 1 else pz + vw
            node_xyzw[v] = (vx, vy, vz, vw)

        node_tri, sz = np.empty((9 * n_sample, 2), np.int32), 0

        def create_sample(n):
            s, t = np.random.random(2)
            if s + t >= 1.0:
                s, t = 1.0 - s, 1.0 - t
            A, B, C = triangles[n]
            return (1 - s - t) * A + s * B + t * C

        def add(v, i):
            nonlocal sz
            node_tri[sz], sz = (v, i), sz + 1

        for i in range(N):
            for _ in range(sample_sz[i]):
                x, y, z = create_sample(i)
                v = 0
                add(v, i)
                for _ in range(8):
                    vx, vy, vz, vw = node_xyzw[v]
                    for r in range(8):
                        c = (v << 3) + r + 1
                        cx, cy, cz, cw = node_xyzw[c]
                        if not (cx - cw <= x <= cx + cw):
                            continue
                        if not (cy - cw <= y <= cy + cw):
                            continue
                        if not (cz - cw <= z <= cz + cw):
                            continue
                        v = c
                        break
                    add(v, i)

        ID = np.argsort(node_tri[:, 1], kind='mergesort')
        node_tri = node_tri[ID]
        ID = np.argsort(node_tri[:, 0], kind='mergesort')
        node_tri = node_tri[ID]
        use = np.ones(len(node_tri), np.bool_)
        for i in range(1, len(node_tri)):
            if node_tri[i - 1, 0] != node_tri[i, 0]:
                continue
            if node_tri[i - 1, 1] != node_tri[i, 1]:
                continue
            use[i] = False
        node_tri = node_tri[use]
        idx = np.searchsorted(node_tri[:, 0], np.arange(maxsize + 1))
        return (triangles, node_xyzw, node_tri, idx)

    @staticmethod
    @njit
    def _nns_from_nodes_to_nodes(
        nodes, octree, k, distance_upper_bound=np.inf
    ):
        points, node_xyzw, node_pt, idx = octree

        def possible_dist_min(node_id, x, y, z):
            nx, ny, nz, nw = node_xyzw[node_id]
            x1 = min(nx + nw, max(nx - nw, x))
            y1 = min(ny + nw, max(ny - nw, y))
            z1 = min(nz + nw, max(nz - nw, z))
            return ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** .5

        def calc_frm(x, y, z):
            # priority queue min dist possible, node_id)
            que = [(0.0, 0)]
            res_q = [(-np.inf, -1)] * k

            while que:
                d, node_id = heapq.heappop(que)
                if d > -res_q[0][0] or d > distance_upper_bound:
                    continue

                if 8 * node_id + 1 < len(node_xyzw):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx[to + 1] - idx[to] == 0:
                            continue
                        d = possible_dist_min(to, x, y, z)
                        heapq.heappush(que, (d, to))
                    continue
                # leaf node
                for point_id in node_pt[idx[node_id]: idx[node_id + 1], 1]:
                    px, py, pz = points[point_id]
                    d = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** .5
                    if d > distance_upper_bound:
                        continue
                    heapq.heappushpop(res_q, (-d, np.int64(point_id)))
            nbd_indices = np.empty(k, np.int32)
            vectors = np.empty((k, 3), np.float64)
            for i in range(k):
                j = heapq.heappop(res_q)[1]
                nbd_indices[i] = j
                if j == -1:
                    vectors[i] = (np.inf, np.inf, np.inf)
                else:
                    vectors[i][0] = points[j, 0] - x
                    vectors[i][1] = points[j, 1] - y
                    vectors[i][2] = points[j, 2] - z
            return nbd_indices[::-1], vectors[::-1]

        N = len(nodes)
        nbd_indices = np.empty((N, k), np.int32)
        vectors = np.empty((N, k, 3), np.float64)
        for i in range(N):
            x, y, z = nodes[i]
            nbd_indices[i], vectors[i] = calc_frm(x, y, z)
        dists = np.empty((N, k), np.float64)
        for i in range(N):
            for n in range(k):
                x, y, z = vectors[i, n]
                dists[i, n] = (x * x + y * y + z * z)**.5
        return nbd_indices, vectors, dists

    @ staticmethod
    @ njit
    def _calc_directed_hausdorff_nodes(
        octree_A, octree_B
    ):
        points_A, node_xyzw_A, node_pt_A, idx_A = octree_A
        points_B, node_xyzw_B, node_pt_B, idx_B = octree_B

        def possible_dist_max_node(i, j):
            dw = node_xyzw_A[i, 3] + node_xyzw_B[j, 3]
            dx = abs(node_xyzw_A[i, 0] - node_xyzw_B[j, 0]) + dw
            dy = abs(node_xyzw_A[i, 1] - node_xyzw_B[j, 1]) + dw
            dz = abs(node_xyzw_A[i, 2] - node_xyzw_B[j, 2]) + dw
            return (dx * dx + dy * dy + dz * dz) ** .5

        def possible_dist_range(node_id, x, y, z):
            nx, ny, nz, nw = node_xyzw_B[node_id]
            dx1 = abs(min(nx + nw, max(nx - nw, x)) - x)
            dy1 = abs(min(ny + nw, max(ny - nw, y)) - y)
            dz1 = abs(min(nz + nw, max(nz - nw, z)) - z)
            dx2 = max(abs(nx - nw - x), abs(nx + nw - x))
            dy2 = max(abs(ny - nw - y), abs(ny + nw - y))
            dz2 = max(abs(nz - nw - z), abs(nz + nw - z))
            lo = (dx1 * dx1 + dy1 * dy1 + dz1 * dz1) ** .5
            hi = (dx2 * dx2 + dy2 * dy2 + dz2 * dz2) ** .5
            return lo, hi

        def calc_frm_node(i):
            que = [(0.0, 0)]
            dist = np.inf
            while que:
                d, node_id = heapq.heappop(que)
                if d > dist:
                    continue
                if 8 * node_id + 1 < len(node_xyzw_B):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx_B[to + 1] - idx_B[to] == 0:
                            continue
                        d = possible_dist_max_node(i, to)
                        if d < dist:
                            heapq.heappush(que, (d, to))
                    continue
                # leaf node
                dist = min(dist, possible_dist_max_node(i, node_id))
            return dist

        que = [(0.0, 0)] * 0
        maxsize = len(node_xyzw_A)
        for i in range(maxsize):
            if 8 * i + 1 < maxsize or idx_A[i + 1] == idx_A[i]:
                continue
            que.append((-calc_frm_node(i), i))
        heapq.heapify(que)

        HD = 0.0

        def calc_frm(x, y, z):
            que = [(0.0, 0)]  # priority queue min dist possible, node_id)
            dist = np.inf

            while que:
                d, node_id = heapq.heappop(que)
                if d > dist:
                    continue

                if 8 * node_id + 1 < len(node_xyzw_B):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx_B[to + 1] - idx_B[to] == 0:
                            continue
                        lo, hi = possible_dist_range(to, x, y, z)
                        if hi <= HD:
                            return 0.0
                        heapq.heappush(que, (lo, to))
                    continue
                # leaf node
                PID = node_pt_B[idx_B[node_id]: idx_B[node_id + 1], 1]
                for point_id in PID:
                    px, py, pz = points_B[point_id]
                    d = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** .5
                    dist = min(dist, d)
            return dist

        while que:
            x, i = heapq.heappop(que)
            dist_upper = -x
            if dist_upper <= HD:
                break
            for point_id in node_pt_A[idx_A[i]:idx_A[i + 1], 1]:
                x, y, z = points_A[point_id]
                HD = max(HD, calc_frm(x, y, z))
        return HD

    @ staticmethod
    @ njit
    def _nns_from_elements_to_nodes(
        tris, octree, k, distance_upper_bound=np.inf
    ):
        N = len(tris)
        n_sample = min(100 * N, 1 << 24)
        areas = np.empty(N)
        for n in range(N):
            A, B, C = tris[n]
            x1, y1, z1 = B - A
            x2, y2, z2 = C - A
            x3 = y1 * z2 - y2 * z1
            y3 = z1 * x2 - z2 * x1
            z3 = x1 * y2 - x2 * y1
            areas[n] = (x3 * x3 + y3 * y3 + z3 * z3)**.5
        areas = np.append(0, np.cumsum(areas))
        sample_sz = (areas * n_sample // areas[-1]).astype(np.int32)
        sample_sz = sample_sz[1:] - sample_sz[:-1]
        sample_sz = np.maximum(sample_sz, 1)

        def distance_triangle_and_point(triangle, point):
            p0, p1, p2 = triangle[0], triangle[1], triangle[2]
            p = point
            n = np.cross(p1 - p0, p2 - p0)
            n /= (n * n).sum()**.5
            n0 = np.cross(n, p2 - p1)
            n1 = np.cross(n, p0 - p2)
            n2 = np.cross(n, p1 - p0)
            side0 = (n0 * (p0 - p1)).sum() * (n0 * (p - p1)).sum() > 0
            side1 = (n1 * (p1 - p2)).sum() * (n1 * (p - p2)).sum() > 0
            side2 = (n2 * (p2 - p0)).sum() * (n2 * (p - p0)).sum() > 0
            if side0 and side1 and side2:
                k = (n * (p - p0)).sum()
                q = p - k * n
                return ((p - q)**2).sum()**.5, q

            min_dist = ((p - p0)**2).sum()**.5
            q = p0

            for _ in range(3):
                p0, p1, p2 = p1, p2, p0
                d = p1 - p0
                t = ((p - p0) * d).sum()
                t /= (d * d).sum()
                if t < 0:
                    t = 0
                elif t > 1:
                    t = 1
                qq = p0 + t * d
                dist = ((p - qq)**2).sum()**.5
                if min_dist > dist:
                    q = qq
                    min_dist = dist
            return min_dist, q

        def create_sample(n):
            s, t = np.random.random(2)
            if s + t >= 1.0:
                s, t = 1.0 - s, 1.0 - t
            A, B, C = tris[n]
            return (1 - s - t) * A + s * B + t * C

        points, node_xyzw, node_pt, idx = octree

        def possible_dist_min(node_id, x, y, z):
            nx, ny, nz, nw = node_xyzw[node_id]
            x1 = min(nx + nw, max(nx - nw, x))
            y1 = min(ny + nw, max(ny - nw, y))
            z1 = min(nz + nw, max(nz - nw, z))
            return ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** .5

        def calc_frm(x, y, z):
            # priority queue min dist possible, frm_pt, node_id)
            que = [(0.0, 0)]
            res_q = [(-np.inf, -1)] * k

            while que:
                d, node_id = heapq.heappop(que)
                if d > -res_q[0][0] or d > distance_upper_bound:
                    continue
                if 8 * node_id + 1 < len(node_xyzw):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx[to + 1] - idx[to] == 0:
                            continue
                        d = possible_dist_min(to, x, y, z)
                        heapq.heappush(que, (d, to))
                    continue
                # leaf node
                for point_id in node_pt[idx[node_id]: idx[node_id + 1], 1]:
                    px, py, pz = points[point_id]
                    d = ((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) ** .5
                    if d > distance_upper_bound:
                        continue
                    heapq.heappushpop(res_q, (-d, np.int64(point_id)))
            return res_q

        nbd_indices = np.empty((N, k), np.int32)
        dists = np.empty((N, k), np.float64)

        done = np.zeros(len(points), np.bool_)

        for n in range(N):
            res_q = [(-np.inf, -1)] * k
            for _ in range(sample_sz[n]):
                x, y, z = create_sample(n)
                for v, i in calc_frm(x, y, z):
                    heapq.heappush(res_q, (v, i))

            cand = [0] * 0
            while res_q:
                if len(cand) == k:
                    break
                i = heapq.heappop(res_q)[1]
                if i == -1 or done[i]:
                    continue
                done[i] = 1
                cand.append(i)
            for i in cand:
                done[i] = 0

            cand += [-1] * (k - len(cand))
            for i in range(k):
                j = cand[i]
                nbd_indices[n, i] = j
                if j == -1:
                    dists[n, i] = np.inf
                else:
                    dists[n, i] = distance_triangle_and_point(
                        tris[n], points[j])[0]
            ID = np.argsort(dists[n])
            dists[n] = dists[n][ID]
            nbd_indices[n] = nbd_indices[n][ID]
        vectors = np.full((N, k, 3), np.inf, np.float64)
        for i in range(N):
            for n in range(k):
                j = nbd_indices[i, n]
                if j == -1:
                    continue
                p = distance_triangle_and_point(tris[i], points[j])[1]
                vectors[i, n] = points[j] - p
        return nbd_indices, vectors, dists

    @ staticmethod
    @ njit
    def _nns_from_nodes_to_elements(
        nodes, octree_e, k, distance_upper_bound=np.inf
    ):
        triangles, node_xyzw, node_tri, idx = octree_e

        def distance_triangle_and_point(triangle, point):
            p0, p1, p2 = triangle[0], triangle[1], triangle[2]
            p = point
            n = np.cross(p1 - p0, p2 - p0)
            n /= (n * n).sum()**.5
            n0 = np.cross(n, p2 - p1)
            n1 = np.cross(n, p0 - p2)
            n2 = np.cross(n, p1 - p0)
            side0 = (n0 * (p0 - p1)).sum() * (n0 * (p - p1)).sum() > 0
            side1 = (n1 * (p1 - p2)).sum() * (n1 * (p - p2)).sum() > 0
            side2 = (n2 * (p2 - p0)).sum() * (n2 * (p - p0)).sum() > 0
            if side0 and side1 and side2:
                k = (n * (p - p0)).sum()
                q = p - k * n
                return ((p - q)**2).sum()**.5, q

            min_dist = ((p - p0)**2).sum()**.5
            q = p0

            for _ in range(3):
                p0, p1, p2 = p1, p2, p0
                d = p1 - p0
                t = ((p - p0) * d).sum()
                t /= (d * d).sum()
                if t < 0:
                    t = 0
                elif t > 1:
                    t = 1
                qq = p0 + t * d
                dist = ((p - qq)**2).sum()**.5
                if min_dist > dist:
                    q = qq
                    min_dist = dist
            return min_dist, q

        def possible_dist_min(node_id, x, y, z):
            nx, ny, nz, nw = node_xyzw[node_id]
            x1 = min(nx + nw, max(nx - nw, x))
            y1 = min(ny + nw, max(ny - nw, y))
            z1 = min(nz + nw, max(nz - nw, z))
            return ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** .5

        def calc_frm(nid):
            x, y, z = nodes[nid]
            # priority queue min dist possible, frm_pt, node_id)
            que = [(0.0, 0)]
            res_q = [(-np.inf, -1)] * k

            done = set([0] * 0)

            while que:
                d, node_id = heapq.heappop(que)
                if d > -res_q[0][0] or d > distance_upper_bound:
                    continue
                if 8 * node_id + 1 < len(node_xyzw):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx[to + 1] - idx[to] == 0:
                            continue
                        d = possible_dist_min(to, x, y, z)
                        heapq.heappush(que, (d, to))
                    continue
                # leaf node
                for tri_id in node_tri[idx[node_id]: idx[node_id + 1], 1]:
                    if tri_id in done:
                        continue
                    done.add(tri_id)
                    d = distance_triangle_and_point(
                        triangles[tri_id], nodes[nid])[0]
                    heapq.heappushpop(res_q, (-d, np.int64(tri_id)))
            return res_q

        N = len(nodes)

        nbd_indices = np.empty((N, k), np.int32)
        dists = np.empty((N, k), np.float64)
        for n in range(N):
            res_q = calc_frm(n)
            for i in range(k):
                x, j = heapq.heappop(res_q)
                nbd_indices[n, i] = j
                dists[n, i] = -x
            ID = np.argsort(dists[n])
            nbd_indices[n] = nbd_indices[n][ID]
            dists[n] = dists[n][ID]
        vectors = np.full((N, k, 3), np.inf, np.float64)
        for i in range(N):
            for n in range(k):
                j = nbd_indices[i, n]
                if j == -1:
                    continue
                p = distance_triangle_and_point(triangles[j], nodes[i])[1]
                vectors[i, n] = p - nodes[i]
        return nbd_indices, vectors, dists

    @ staticmethod
    @ njit
    def _nns_from_elements_to_elements(
        tris, octree_e, k, distance_upper_bound=np.inf
    ):
        triangles, node_xyzw, node_tri, idx = octree_e

        N = len(tris)
        n_sample = min(100 * N, 1 << 24)
        areas = np.empty(N)
        for n in range(N):
            A, B, C = tris[n]
            x1, y1, z1 = B - A
            x2, y2, z2 = C - A
            x3 = y1 * z2 - y2 * z1
            y3 = z1 * x2 - z2 * x1
            z3 = x1 * y2 - x2 * y1
            areas[n] = (x3 * x3 + y3 * y3 + z3 * z3)**.5
        areas = np.append(0, np.cumsum(areas))
        sample_sz = (areas * n_sample // areas[-1]).astype(np.int32)
        sample_sz = sample_sz[1:] - sample_sz[:-1]
        sample_sz = np.maximum(sample_sz, 1)

        def create_sample(n):
            s, t = np.random.random(2)
            if s + t >= 1.0:
                s, t = 1.0 - s, 1.0 - t
            A, B, C = tris[n]
            return (1 - s - t) * A + s * B + t * C

        def distance_triangle_and_point(triangle, point):
            p0, p1, p2 = triangle[0], triangle[1], triangle[2]
            p = point
            n = np.cross(p1 - p0, p2 - p0)
            n /= (n * n).sum()**.5
            n0 = np.cross(n, p2 - p1)
            n1 = np.cross(n, p0 - p2)
            n2 = np.cross(n, p1 - p0)
            side0 = (n0 * (p0 - p1)).sum() * (n0 * (p - p1)).sum() > 0
            side1 = (n1 * (p1 - p2)).sum() * (n1 * (p - p2)).sum() > 0
            side2 = (n2 * (p2 - p0)).sum() * (n2 * (p - p0)).sum() > 0
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

        def hausdorff_dist_tri(tri1, tri2):
            HD = 0.0
            for _ in range(2):
                tri1, tri2 = tri2, tri1
                for i in range(3):
                    p = tri1[i]
                    HD = max(HD, distance_triangle_and_point(tri2, p))
            return HD

        def possible_dist_min(node_id, x, y, z):
            nx, ny, nz, nw = node_xyzw[node_id]
            x1 = min(nx + nw, max(nx - nw, x))
            y1 = min(ny + nw, max(ny - nw, y))
            z1 = min(nz + nw, max(nz - nw, z))
            return ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** .5

        def calc_frm(x, y, z):
            # priority queue min dist possible, frm_pt, node_id)
            que = [(0.0, 0)]
            res_q = [(-np.inf, -1)] * k

            done = set([0] * 0)

            while que:
                d, node_id = heapq.heappop(que)
                if d > -res_q[0][0] or d > distance_upper_bound:
                    continue
                if 8 * node_id + 1 < len(node_xyzw):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx[to + 1] - idx[to] == 0:
                            continue
                        d = possible_dist_min(to, x, y, z)
                        heapq.heappush(que, (d, to))
                    continue
                # leaf node
                for tri_id in node_tri[idx[node_id]: idx[node_id + 1], 1]:
                    if tri_id in done:
                        continue
                    done.add(tri_id)
                    d = distance_triangle_and_point(
                        triangles[tri_id], np.array([x, y, z]))
                    heapq.heappushpop(res_q, (-d, np.int64(tri_id)))
            return res_q

        nbd_indices = np.empty((N, k), np.int32)
        dists = np.empty((N, k), np.float64)

        done = np.zeros(len(triangles), np.bool_)

        for n in range(N):
            res_q = [(-np.inf, -1)] * k
            for _ in range(sample_sz[n]):
                x, y, z = create_sample(n)
                for v, i in calc_frm(x, y, z):
                    heapq.heappush(res_q, (v, i))

            cand = [0] * 0
            while res_q:
                if len(cand) == k:
                    break
                i = heapq.heappop(res_q)[1]
                if i == -1 or done[i]:
                    continue
                done[i] = 1
                cand.append(i)
            for i in cand:
                done[i] = 0

            cand += [-1] * (k - len(cand))
            for i in range(k):
                j = cand[i]
                nbd_indices[n, i] = j
                if j == -1:
                    dists[n, i] = np.inf
                else:
                    dists[n, i] = hausdorff_dist_tri(tris[n], triangles[j])
            ID = np.argsort(dists[n])
            dists[n] = dists[n][ID]
            nbd_indices[n] = nbd_indices[n][ID]
        return nbd_indices, dists

    @ staticmethod
    @ njit
    def _calc_directed_hausdorff_elements(
        octree_A, octree_B
    ):
        tris_A, node_xyzw_A, node_tri_A, idx_A = octree_A
        tris_B, node_xyzw_B, node_tri_B, idx_B = octree_B

        N = len(tris_A)
        n_sample = min(100 * N, 1 << 24)
        areas = np.empty(N)
        for n in range(N):
            A, B, C = tris_A[n]
            x1, y1, z1 = B - A
            x2, y2, z2 = C - A
            x3 = y1 * z2 - y2 * z1
            y3 = z1 * x2 - z2 * x1
            z3 = x1 * y2 - x2 * y1
            areas[n] = (x3 * x3 + y3 * y3 + z3 * z3)**.5
        areas = np.append(0, np.cumsum(areas))
        sample_sz = (areas * n_sample // areas[-1]).astype(np.int32)
        sample_sz = sample_sz[1:] - sample_sz[:-1]
        sample_sz = np.maximum(sample_sz, 1)

        def create_sample(n):
            s, t = np.random.random(2)
            if s + t >= 1.0:
                s, t = 1.0 - s, 1.0 - t
            A, B, C = tris_A[n]
            return (1 - s - t) * A + s * B + t * C

        def distance_triangle_and_point(triangle, point):
            p0, p1, p2 = triangle[0], triangle[1], triangle[2]
            p = point
            n = np.cross(p1 - p0, p2 - p0)
            n /= (n * n).sum()**.5
            n0 = np.cross(n, p2 - p1)
            n1 = np.cross(n, p0 - p2)
            n2 = np.cross(n, p1 - p0)
            side0 = (n0 * (p0 - p1)).sum() * (n0 * (p - p1)).sum() > 0
            side1 = (n1 * (p1 - p2)).sum() * (n1 * (p - p2)).sum() > 0
            side2 = (n2 * (p2 - p0)).sum() * (n2 * (p - p0)).sum() > 0
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

        def possible_dist_max_node(i, j):
            dw = node_xyzw_A[i, 3] + node_xyzw_B[j, 3]
            dx = abs(node_xyzw_A[i, 0] - node_xyzw_B[j, 0]) + dw
            dy = abs(node_xyzw_A[i, 1] - node_xyzw_B[j, 1]) + dw
            dz = abs(node_xyzw_A[i, 2] - node_xyzw_B[j, 2]) + dw
            return (dx * dx + dy * dy + dz * dz) ** .5

        def possible_dist_range(node_id, x, y, z):
            nx, ny, nz, nw = node_xyzw_B[node_id]
            dx1 = abs(min(nx + nw, max(nx - nw, x)) - x)
            dy1 = abs(min(ny + nw, max(ny - nw, y)) - y)
            dz1 = abs(min(nz + nw, max(nz - nw, z)) - z)
            dx2 = max(abs(nx - nw - x), abs(nx + nw - x))
            dy2 = max(abs(ny - nw - y), abs(ny + nw - y))
            dz2 = max(abs(nz - nw - z), abs(nz + nw - z))
            lo = (dx1 * dx1 + dy1 * dy1 + dz1 * dz1) ** .5
            hi = (dx2 * dx2 + dy2 * dy2 + dz2 * dz2) ** .5
            return lo, hi

        def calc_frm_node(i):
            que = [(0.0, 0)]
            dist = np.inf
            while que:
                d, node_id = heapq.heappop(que)
                if d > dist:
                    continue
                if 8 * node_id + 1 < len(node_xyzw_B):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx_B[to + 1] - idx_B[to] == 0:
                            continue
                        d = possible_dist_max_node(i, to)
                        if d < dist:
                            heapq.heappush(que, (d, to))
                    continue
                # leaf node
                dist = min(dist, possible_dist_max_node(i, node_id))
            return dist

        que = [(0.0, 0)] * 0
        maxsize = len(node_xyzw_A)
        for i in range(maxsize):
            if 8 * i + 1 < maxsize or idx_A[i + 1] == idx_A[i]:
                continue
            que.append((-calc_frm_node(i), i))
        heapq.heapify(que)

        HD = 0.0

        def calc_frm(x, y, z):
            que = [(0.0, 0)]  # priority queue min dist possible, node_id)
            dist = np.inf

            while que:
                d, node_id = heapq.heappop(que)
                if d > dist:
                    continue

                if 8 * node_id + 1 < len(node_xyzw_B):
                    for r in range(8):
                        to = 8 * node_id + r + 1
                        if idx_B[to + 1] - idx_B[to] == 0:
                            continue
                        lo, hi = possible_dist_range(to, x, y, z)
                        if hi <= HD:
                            return 0.0
                        heapq.heappush(que, (lo, to))
                    continue
                # leaf node
                TID = node_tri_B[idx_B[node_id]: idx_B[node_id + 1], 1]
                for tri_id in TID:
                    p = np.array([x, y, z])
                    d = distance_triangle_and_point(tris_B[tri_id], p)
                    dist = min(dist, d)
            return dist
        best_i = -1

        while que:
            x, i = heapq.heappop(que)
            dist_upper = -x
            if dist_upper <= HD:
                break
            TID = node_tri_A[idx_A[i]:idx_A[i + 1], 1]
            for tri_id in TID:
                x, y, z = create_sample(tri_id)
                dist = calc_frm(x, y, z)
                if HD < dist:
                    HD = dist
                    best_i = tri_id
        for _ in range(10 ** 5):
            x, y, z = create_sample(best_i)
            HD = max(HD, calc_frm(x, y, z))
        return HD

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

        dists : np.ndarray
            2D array of shape (N, k) containing float data.
            The length of vectors.
        """
        if target_fem_data is None:
            target_fem_data = self

        points = target_fem_data.nodes.data
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        zmin = points[:, 2].min()
        zmax = points[:, 2].max()
        boundingbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        octree = self.build_octree_node(points, boundingbox)
        nodes = self.nodes.data
        return self._nns_from_nodes_to_nodes(
            nodes, octree, k, distance_upper_bound)

    def nearest_neighbor_search_from_elements_to_nodes(
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

        dists : np.ndarray
            2D array of shape (N, k) containing float data.
            The length of vectors.
        """
        if target_fem_data is None:
            target_fem_data = self

        points = target_fem_data.nodes.data
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        zmin = points[:, 2].min()
        zmax = points[:, 2].max()
        boundingbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        octree = self.build_octree_node(points, boundingbox)

        tris = self.elements.data
        N = len(tris)
        tris = self.collect_node_positions_by_ids(
            tris.ravel()).reshape(N, 3, 3)
        return self._nns_from_elements_to_nodes(
            tris, octree, k, distance_upper_bound)

    def nearest_neighbor_search_from_nodes_to_elements(
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

        dists : np.ndarray
            2D array of shape (N, k) containing float data.
            The length of vectors.
        """
        if target_fem_data is None:
            target_fem_data = self

        tris = target_fem_data.elements.data
        N = len(tris)
        tris = target_fem_data.collect_node_positions_by_ids(
            tris.ravel()).reshape(N, 3, 3)
        points = target_fem_data.nodes.data
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        zmin = points[:, 2].min()
        zmax = points[:, 2].max()
        boundingbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        octree_e = self.build_octree_element(tris, boundingbox)
        return self._nns_from_nodes_to_elements(
            self.nodes.data, octree_e, k, distance_upper_bound)

    def nearest_neighbor_search_from_elements_to_elements(
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

        dists : np.ndarray
            2D array of shape (N, k) containing float data.
            Distances from the elements to nearest points.
            If the number of points found is less than k,
            corresponding values are filled by np.inf.
        """
        if target_fem_data is None:
            target_fem_data = self

        tris = target_fem_data.elements.data
        N = len(tris)
        tris = target_fem_data.collect_node_positions_by_ids(
            tris.ravel()).reshape(N, 3, 3)
        points = target_fem_data.nodes.data
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        zmin = points[:, 2].min()
        zmax = points[:, 2].max()
        boundingbox = (xmin, xmax, ymin, ymax, zmin, zmax)
        octree_e = self.build_octree_element(tris, boundingbox)

        tris = self.elements.data
        N = len(tris)
        tris = self.collect_node_positions_by_ids(
            tris.ravel()).reshape(N, 3, 3)

        return self._nns_from_elements_to_elements(
            tris, octree_e, k, distance_upper_bound)

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
        A = self.nodes.data
        B = target_fem_data.nodes.data
        xmin = min(A[:, 0].min(), B[:, 0].min())
        xmax = max(A[:, 0].max(), B[:, 0].max())
        ymin = min(A[:, 1].min(), B[:, 1].min())
        ymax = max(A[:, 1].max(), B[:, 1].max())
        zmin = min(A[:, 2].min(), B[:, 2].min())
        zmax = max(A[:, 2].max(), B[:, 2].max())
        boundingbox = (xmin, xmax, ymin, ymax, zmin, zmax)

        octree_A = self.build_octree_node(A, boundingbox)
        octree_B = self.build_octree_node(B, boundingbox)

        if directed:
            return self._calc_directed_hausdorff_nodes(octree_A, octree_B)
        else:
            HD1 = self._calc_directed_hausdorff_nodes(octree_A, octree_B)
            HD2 = self._calc_directed_hausdorff_nodes(octree_B, octree_A)
            return max(HD1, HD2)

    def calculate_hausdorff_distance_elements(
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
        A = self.nodes.data
        B = target_fem_data.nodes.data
        xmin = min(A[:, 0].min(), B[:, 0].min())
        xmax = max(A[:, 0].max(), B[:, 0].max())
        ymin = min(A[:, 1].min(), B[:, 1].min())
        ymax = max(A[:, 1].max(), B[:, 1].max())
        zmin = min(A[:, 2].min(), B[:, 2].min())
        zmax = max(A[:, 2].max(), B[:, 2].max())
        boundingbox = (xmin, xmax, ymin, ymax, zmin, zmax)

        tris_A = self.elements.data
        N = len(tris_A)
        tris_A = self.collect_node_positions_by_ids(
            tris_A.ravel()).reshape(N, 3, 3)

        tris_B = target_fem_data.elements.data
        N = len(tris_B)
        tris_B = target_fem_data.collect_node_positions_by_ids(
            tris_B.ravel()).reshape(N, 3, 3)

        octree_A = self.build_octree_element(tris_A, boundingbox)
        octree_B = self.build_octree_element(tris_B, boundingbox)

        if directed:
            return self._calc_directed_hausdorff_elements(octree_A, octree_B)
        else:
            HD1 = self._calc_directed_hausdorff_elements(octree_A, octree_B)
            HD2 = self._calc_directed_hausdorff_elements(octree_B, octree_A)
            return max(HD1, HD2)

import datetime as dt
import functools

import networkx as nx
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

        Returns:
            surface_indices: indices of nodes (not IDs).
            surface_positions: Positions of each nodes on the surface.
        """
        dict_facets = self.extract_facets()
        dict_facet_shapes = {'tri': [], 'quad': []}
        for facet in dict_facets.values():
            n_node_per_element = facet.shape[-1]
            if n_node_per_element == 3:
                dict_facet_shapes['tri'].append(facet)
            elif n_node_per_element == 4:
                dict_facet_shapes['quad'].append(facet)
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

        Returns:
            surface_data: 2D array of int.
            row data correspond to (element_id, surface_id) of surface.
        """
        data = self.elements.data
        N = len(data)

        # node_0, node_1, node_2, elem_id, surf_id
        surfs = np.empty((4*N, 5), np.int32)
        surfs[0*N:1*N, :3] = data[:, [0, 1, 2]]
        surfs[1*N:2*N, :3] = data[:, [0, 1, 3]]
        surfs[2*N:3*N, :3] = data[:, [1, 2, 3]]
        surfs[3*N:4*N, :3] = data[:, [2, 0, 3]]
        surfs[0*N:1*N, 3] = self.elements.ids
        surfs[1*N:2*N, 3] = self.elements.ids
        surfs[2*N:3*N, 3] = self.elements.ids
        surfs[3*N:4*N, 3] = self.elements.ids
        surfs[0*N:1*N, 4] = 1
        surfs[1*N:2*N, 4] = 2
        surfs[2*N:3*N, 4] = 3
        surfs[3*N:4*N, 4] = 4

        surfs[:, :3].sort(axis=1)
        ind = np.lexsort(
            (surfs[:, 4], surfs[:, 3], surfs[:, 2], surfs[:, 1], surfs[:, 0]))
        surfs = surfs[ind]

        # select surce
        unique = np.ones(4*N, np.bool_)
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
        facets: Dict[numpy.ndarray]
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
            facets = elements.data
        else:
            facets = self._generate_all_faces(
                elements, element_type, method=method)

        if remove_duplicates:
            facets = functions.remove_duplicates(facets)

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

        return face_ids

    @functools.lru_cache(maxsize=1)
    def filter_first_order_nodes(self):
        """Obtain filter to get first order nodes.

        Args:
            None
        Returns:
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

        Returns:
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

        Args:
            mode: str, optional (['elemental'], 'nodal')
                If 'elemental', generate (n_element, n_element) shaped
                adjacency matrix where edges are defined by node shearing.
                If 'nodal', generate (n_node, n_node) shaped adjacency matrix
                with edges are defined by element shearing.
            order1_only: bool, optional [True]
                If True, consider only order 1 nodes. Effective only when
                mode == 'nodal'.

        Returns:
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

        Returns:
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
        edges.

        Args:
            order1_only: bool, optional [True]
                If True, consider only order 1 nodes.
        Returns:
            adj: scipy.sparse.csr_matrix
                Adjacency matrix in CSR expression.
            node2nodes: dict
                Dictionary of node ID to adjacent node IDs.
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
                [True]*len(element_indices),
                (np.concatenate(node_indices), element_indices)),
            shape=(len(nodes), len(elements)))
        return incidence_matrix

    @functools.lru_cache(maxsize=15)
    def calculate_n_hop_adj(
            self, mode='elemental', n_hop=1, include_self_loop=False,
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

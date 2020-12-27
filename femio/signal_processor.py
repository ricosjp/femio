import datetime as dt
import functools

import numpy as np
import scipy.sparse as sp

from .fem_attribute import FEMAttribute


class SignalProcessorMixin:

    def convert_nodal2elemental(self, data, *,
                                calc_average=False, ravel=False):
        """Convert nodal data to elemental data.

        Args:
            data: String or numpy.ndarray
                Name of nodal data or nodal data itself.
            calc_average: Bool, optional [False]
                If True, output is averaged element by element.
            ravel: Bool, optional [False]
                If True, output is raveled to make 2D array.
        Returns:
            elemental_data: numpy.ndarray
        """
        if isinstance(data, str):
            name_data = data
            nodal_data = self.nodal_data.get_attribute_data(name_data)
        else:
            nodal_data = data
        if len(nodal_data) != len(self.nodes.ids):
            raise ValueError(
                f"Input nodal data length is not the same as that of node")

        elemental_data = np.array([
            nodal_data[
                self.nodes.ids2indices(nodes), :]
            for nodes in self.elements.data])
        if calc_average:
            return np.mean(elemental_data, axis=1)
        if ravel:
            return np.array([np.ravel(r) for r in elemental_data])
        return elemental_data

    def convert_elemental2nodal(
            self, elemental_data, mode='mean', order1_only=True,
            raise_negative_volume=True):
        """Convert elemental data to nodal data.

        Args:
            elemental_data: numpy.ndarray
                Elemental data to be converted.
            mode: str, optional (['effective'], 'mean')
                The way haw to convert. 'effective' means weighted integration
                which results in consistent volume. 'mean' means weighted
                mean which results in smoother field at the boundary but
                not necessarily consistent volume.
            order1_only: bool, optional [True]
                If True, convert data only for order 1 nodes.
            raise_negative_volume: bool, optional [True]
                If True, raise ValueError when negative volume found.
        Returns:
            converted_data: numpy.ndarray
        """
        if len(elemental_data) != len(self.elements.ids):
            raise ValueError(
                'Length of input data differs from that of elements')
        incidence_matrix = self.calculate_incidence_matrix(
            order1_only=order1_only)

        if mode == 'effective':
            if not order1_only:
                raise NotImplementedError
            weighted_incidence_matrix = incidence_matrix.multiply(
                1 / incidence_matrix.sum(axis=0))

        elif mode == 'mean':
            volumes = self.calculate_element_volumes(
                raise_negative_volume=raise_negative_volume)
            volume_incidence_matrix = incidence_matrix.multiply(volumes.T)
            weighted_incidence_matrix = volume_incidence_matrix.multiply(
                1 / volume_incidence_matrix.sum(axis=1))

        else:
            raise ValueError(f"Invalid mode: {mode}")

        converted_data = weighted_incidence_matrix.dot(elemental_data)

        return converted_data

    def convert_lte_global2local(self):
        """Convert linear thermal expansion coefficient from global to local
        point of view.
        """
        lte_full = self.elemental_data.get_attribute_data('lte_full')
        lte_mats = np.reshape(np.array([
            [lte_full[:, 0], lte_full[:, 3] / 2, lte_full[:, 5] / 2.,
             lte_full[:, 3] / 2., lte_full[:, 1], lte_full[:, 4] / 2.,
             lte_full[:, 5] / 2., lte_full[:, 4] / 2., lte_full[:, 2],
             ]
        ]).T, (-1, 3, 3))
        ws, vs = np.linalg.eigh(lte_mats)
        orients = np.concatenate(
            [np.reshape(np.transpose(vs, (0, 2, 1)), (-1, 9))[:, :6],
             np.zeros((len(self.elements.ids), 3))],
            axis=1)  # Omit the third axis to be right-handed system

        self.elemental_data.update({
            'linear_thermal_expansion_coefficient':
            FEMAttribute(
                'linear_thermal_expansion_coefficient',
                self.elements.ids, ws)})
        self.elemental_data.update({
            'ORIENTATION':
            FEMAttribute(
                'ORIENTATION',
                self.elements.ids, orients)})
        return

    def convert_lte_local2global(self):
        """Convert linear thermal expansion coefficient from local to global
        point of view.
        """
        local_ltes = self.elemental_data.get_attribute_data('lte')
        diag_ltes = np.array([np.diag(lte) for lte in local_ltes])
        orients = self.elemental_data.get_attribute_data('orient')
        orients = np.stack(
            [orients[:, 0:3], orients[:, 3:6],
             np.cross(orients[:, 0:3], orients[:, 3:6])],
            axis=1)
        lte_mat = np.matmul(
            np.transpose(orients, (0, 2, 1)),
            np.matmul(diag_ltes, orients))
        lte_full = np.stack(
            [lte_mat[:, 0, 0], lte_mat[:, 1, 1], lte_mat[:, 2, 2],
             lte_mat[:, 0, 1] * 2, lte_mat[:, 1, 2] * 2, lte_mat[:, 0, 2] * 2],
            axis=1)

        self.elemental_data.update(
            {'linear_thermal_expansion_coefficient_full':
             FEMAttribute(
                 'linear_thermal_expansion_coefficient_full',
                 self.elements.ids, lte_full)
             }
        )

        self.material_overwritten = True
        return

    def invert_strain(self, strain, is_engineering=False):
        values, directions, vectors = self.calculate_principal_components(
            strain, from_engineering=is_engineering)
        # inverted_values = np.exp(- np.log(1 + values)) - 1
        inverted_values = 1. / (1. + values) - 1.
        return self.calculate_array_from_eigens(
            inverted_values, directions, to_engineering=is_engineering)

    def calculate_array_from_eigens(
            self, eigenvalues, eigenvectors, to_engineering=False):
        return self.convert_symmetric_matrix2array(
            self.calculate_symmetric_matrices_from_eigens(
                eigenvalues, eigenvectors), to_engineering=to_engineering)

    def calculate_symmetric_matrices_from_eigens(
            self, eigenvalues, eigenvectors):
        diags = self.convert_array2symmetric_matrix(np.concatenate(
            [eigenvalues, np.zeros(eigenvalues.shape)], axis=1))
        rotation_matrices = np.stack([
            eigenvectors[:, :3],
            eigenvectors[:, 3:6],
            eigenvectors[:, 6:],
        ], axis=2)
        return rotation_matrices @ diags @ np.transpose(
            rotation_matrices, (0, 2, 1))

    def add_principal_vectors(self, name_variable):
        if name_variable in [
                'lte_full', 'linear_thermal_expansion_coefficient_full']:
            values, _, vectors = self.calculate_principal_components(
                self.elemental_data.get_attribute_data('lte_full'),
                from_engineering=True)
        elif name_variable in [
                'elemental_strain', 'ElementalSTRAIN']:
            values, _, vectors = self.calculate_principal_components(
                self.elemental_data.get_attribute_data('elemental_strain'),
                from_engineering=True)
        elif name_variable in ['fiber_orientation_tensor']:
            values, _, vectors = self.calculate_principal_components(
                self.elemental_data.get_attribute_data(
                    'fiber_orientation_tensor'),
                from_engineering=False, order=[0, 3, 5, 1, 4, 2])
        else:
            raise ValueError(f"Unknown name_variable: {name_variable}")

        self.elemental_data.update(
            {
                f"principal_{name_variable}_1":
                FEMAttribute(
                    f"principal_{name_variable}_1",
                    self.elements.ids, vectors[:, :3]),
                f"principal_{name_variable}_2":
                FEMAttribute(
                    f"principal_{name_variable}_2",
                    self.elements.ids, vectors[:, 3:6]),
                f"principal_{name_variable}_3":
                FEMAttribute(
                    f"principal_{name_variable}_3",
                    self.elements.ids, vectors[:, 6:]),
                f"principal_{name_variable}_value_1":
                FEMAttribute(
                    f"principal_{name_variable}_value_1",
                    self.elements.ids, values[:, 0]),
                f"principal_{name_variable}_value_2":
                FEMAttribute(
                    f"principal_{name_variable}_value_2",
                    self.elements.ids, values[:, 1]),
                f"principal_{name_variable}_value_3":
                FEMAttribute(
                    f"principal_{name_variable}_value_3",
                    self.elements.ids, values[:, 2]),
             }
        )

    def calculate_principal_components(
            self, in_array, *,
            from_engineering=False, order=None):
        """Calculate eigenvalues and eigenvectors of the input arrays which are
        parts of symmetric matrices.

        Args:
            in_array: numpy.ndarray
                (n, 6) shaped array to form symmetric matrices.
            from_engineering: bool, optional [False]
                If True, treat in_array as engineering-strain-like data.
            order: array-like, optional [[0, 1, 2, 3, 4, 5]]
                The order of in_array. The order should be
                [11, 22, 33, 12, 23, 31] order.
        Returns:
            eigenvalues: numpy.ndarray
                (n, 3) shaped array of eigenvalues.
            eigenvectors: numpy.ndarray
                (n, 9) shaped array of eigenvectors.
        """
        symmetrc_matrix = self.convert_array2symmetric_matrix(
            in_array, from_engineering=from_engineering, order=order)
        _eigenvalues, _directions = np.linalg.eigh(symmetrc_matrix)

        # Sort descending order
        eigenvalues = _eigenvalues[:, ::-1]
        directions = _directions[:, :, ::-1]

        # Make sure it right handed system
        directions[:, :, 2] = np.cross(
            directions[:, :, 0], directions[:, :, 1])

        vectors = np.einsum(
            'ik,ijk->ijk', eigenvalues, directions)
        return (
            eigenvalues,
            np.concatenate([
                directions[:, :, 0], directions[:, :, 1],
                directions[:, :, 2]], axis=1),
            np.concatenate([
                vectors[:, :, 0], vectors[:, :, 1],
                vectors[:, :, 2]], axis=1))

    def convert_array2symmetric_matrix(
            self, in_array, *,
            from_engineering=False, order=None):
        """Convert (n, 6) shaped array to (n, 3, 3) symmetric matrix.

        Args:
            in_array: numpy.ndarray
                (n, 6) shaped array.
            from_engineering: bool, optional [False]
                If True, treat in_array as engineering-strain-like data.
            order: array-like, optional [[0, 1, 2, 3, 4, 5]]
                The order of in_array. The order should be
                [11, 22, 33, 12, 23, 31] order.
        Returns:
            symmetrc_matrix: numpy.ndarray
                (n, 6, 6) symmetric matrix.
        """
        if order is None:
            order = [0, 1, 2, 3, 4, 5]
        in_array = in_array[:, order]
        if from_engineering:
            # Handle engineering strain stuff
            in_array[:, 3:] = in_array[:, 3:] / 2

        indices_array2symmetric_matrix = [0, 3, 5, 3, 1, 4, 5, 4, 2]
        return np.reshape(
            in_array[:, indices_array2symmetric_matrix], (-1, 3, 3))

    def convert_symmetric_matrix2array(
            self, in_matrix, *,
            to_engineering=False, order=None):
        """Convert (n, 3, 3) shaped symmetric matrices to (n, 6) array.

        Args:
            in_matrix: numpy.ndarray
                (n, 6, 6) symmetric matrix.
            to_engineering: bool, optional [False]
                If True, the out_array will be converted to
                engineering-strain-like data.
            order: array-like, optional [[0, 1, 2, 3, 4, 5]]
                The order of out_array. The order should be
                [11, 22, 33, 12, 23, 31] order.
        Returns:
            out_array: numpy.ndarray
                (n, 6) shaped array.
        """
        if order is None:
            order = [0, 1, 2, 3, 4, 5]

        indices_symmetric_matrix2array = [0, 4, 8, 1, 5, 2]
        out_array = np.reshape(in_matrix, (-1, 9))[
            :, indices_symmetric_matrix2array]

        if to_engineering:
            # Handle engineering strain stuff
            out_array[:, 3:] = out_array[:, 3:] * 2
        return out_array[:, order]

    def integrate_elements(self, nodal_data):
        """Integrate nodal data with element volumes.

        Args:
        nodal_data: np.ndarray
            Nodal data with (n_nodes, n_features) shape.

        Returns:
        integrated_elements: np.ndarray
            Integrated elements data with (n_elements, n_features) shape.
        """
        volumes = self.calculate_element_volumes()
        if self.elements.element_type == 'tet':
            nodal_data_by_elements = np.mean(nodal_data[np.array([
                self.collect_node_indices_by_ids(e)
                for e in self.elements.data])], axis=1)
            return nodal_data_by_elements * volumes
        else:
            raise NotImplementedError

    def integrate(self, nodal_data):
        """Integrate nodal data over element volumes.

        Args:
        nodal_data: np.ndarray
            Nodal data with (n_nodes, n_features) shape.

        Returns:
        integrated_elements: np.ndarray
            Integrated elements data with (n_features,) shape.
        """
        return np.sum(self.integrate_elements(nodal_data), axis=0)

    def calculate_diffusion_elemental_data(
            self, elemental_data, *, hops=1, weight=1e-4):
        """Perform diffusion of elemental data according with graph Laplacian.

        Args:
            elemental_data: numpy.ndarray
                Elemental data to be moiving-averaged.
            hops: int, optional [1]
                The number of hops to make moving averaging.
            weight: float, optional [1e-4]
                Weight to be used in diffusion (= diffusion coeff).
        Returns:
            diffused_elemental_data: numpy.ndarray
                The results of diffusion.
        """
        if hops == 0:
            return elemental_data

        adj = self.calculate_adjacency_matrix_element()
        adj_wo_loop = adj - sp.eye(len(self.elements.ids))

        # raise ValueError(adj, np.array(adj_wo_loop.sum(1)).flatten())
        degree_matrix = sp.diags(np.array(adj_wo_loop.sum(1)).flatten())
        laplacian_matrix = adj_wo_loop - degree_matrix

        print('Calculating graph diffusion')
        print(dt.datetime.now())
        diffused_data = elemental_data
        for _ in range(hops):
            # NOTE: Don't use += since timing will be inconsistent
            diffused_data = diffused_data + weight * laplacian_matrix.dot(
                diffused_data)

        return diffused_data

    def calculate_moving_average_elemental_data(
            self, elemental_data, *, hops=1, weight=.1):
        """Perform moving average according with adjacency of the mesh.

        Args:
            elemental_data: numpy.ndarray
                Elemental data to be moiving-averaged.
            hops: int, optional [1]
                The number of hops to make moving averaging.
            weight: float, optional [.1]
                Weight to be used in moving averaging.
        Returns:
            moving_averaged_elemental_data: numpy.ndarray
                The results of moving averaging.
        """
        if hops == 0:
            return elemental_data

        adj = self.calculate_adjacency_matrix_element()
        adj_wo_loop = adj - sp.eye(len(self.elements.ids))
        normalizers = 1 / (1 + weight * np.array(adj_wo_loop.sum(1)))

        print('Calculating moving average')
        print(dt.datetime.now())
        averaged_data = elemental_data
        for _ in range(hops):
            averaged_data = normalizers * (
                averaged_data + weight * adj_wo_loop.dot(averaged_data))
        return averaged_data

    def calculate_moving_average_nodal_data(
            self, nodal_data, *, hops=1, weight=.1, order1_only=True):
        """Perform moving average according with adjacency of the mesh.

        Args:
            nordal_data: numpy.ndarray
                Nodal data to be moiving-averaged.
            hops: int, optional [1]
                The number of hops to make moving averaging.
            weight: float, optional [.1]
                Weight to be used in moving averaging.
            order1_only: bool, optional [True]
                If True, nordal data should be associated with order 1 nodes.
        Returns:
            moving_averaged_nodal_data: numpy.ndarray
                The results of moving averaging.
        """
        if hops == 0:
            return nodal_data
        if order1_only is True:
            if self.elements.element_type == 'tet2':
                node_ids = self.nodes.ids[self.filter_first_order_nodes()]
            elif self.elements.element_type in ['tri', 'quad', 'tet']:
                node_ids = self.nodes.ids
            else:
                raise NotImplementedError
        else:
            node_ids = self.nodes.ids

        adj = self.calculate_adjacency_matrix_node(order1_only=order1_only)
        adj_wo_loop = adj - sp.eye(len(node_ids))
        normalizers = 1 / (1 + weight * np.array(adj_wo_loop.sum(1)))

        print('Calculating moving average')
        print(dt.datetime.now())
        averaged_data = nodal_data
        for _ in range(hops):
            averaged_data = normalizers * (
                averaged_data + weight * adj_wo_loop.dot(averaged_data))
        return averaged_data

    def calculate_median_filter(
            self, data, *, mode='elemental', hops=1, order1_only=True):
        """Perform median filter according with adjacency of the mesh.

        Args:
            data: numpy.ndarray
                Data to be filtered.
            hops: int, optional [1]
                The number of hops to make filtering.
            mode: str, optional, (['elemental'], 'nodal')
            order1_only: bool, optional [True]
                If True, nordal data should be associated with order 1 nodes.
        Returns:
            moving_averaged_nodal_data: numpy.ndarray
                The results of filtering.
        """
        if hops == 0:
            return data

        adj = self.calculate_adjacency_matrix(
            mode=mode, order1_only=order1_only).tocsr()
        list_adj = [a.tocoo().col for a in adj]

        print('Applying median filter')
        print(dt.datetime.now())
        filtered_data = data
        for _ in range(hops):
            temp_data = np.zeros(data.shape)
            for i_row, adjacents in enumerate(list_adj):
                temp_data[i_row, :] = np.median(
                    filtered_data[adjacents], axis=0)
            filtered_data = temp_data
        return filtered_data

    def calculate_frame_expansion_adjs(
            self, data, mode='nodal', n_hop=1, order1_only=True, **kwargs):
        """Calculate frame expansion adjacency matrices.

        Parameters
        ----------
        data: numpy.ndarray
            (n_node, dimension, dimension, ...) shaped array to be expanded.
                     ^^^^^^^^^^^^^^^^^^^^^^^^^
                     k repetitions for rank k tensor
        mode: ['nodal', 'elemental'], optional
            If 'nodal', deal with nodal data. If 'elemental', deal with
            elemental data. The default is 'nodal'.
        n_hop: int, optional
            The number of hops to be considered. The default is 1.
        order1_only: bool, optional
            If True, take into account only order 1 nodes. The default is True.

        Returns
        -------
        frame_expansion_matric: List[scipy.sparse.csr_matrix]
            List of shape (dimension, dimension, ...) which contains
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          k repetitions for rank k tensor
            (n_node, n_node) shaped sparse matrices.
        """
        # TOD: Implement after tensor class is prepared
        raise NotImplementedError
        tensor_rank = len(data.shape) - 1
        if tensor_rank > 2:
            raise NotImplementedError
        if tensor_rank < 1:
            raise ValueError(f"Invalid input data shape: {data.shape}")

        frame_tensor_adjs \
            = self.calculate_frame_tensor_adjs(
                mode=mode, n_hop=n_hop, order1_only=order1_only,
                tensor_rank=tensor_rank)

        frame_expansion_adjs = self._dot_ndarray_sparse(
            data, frame_tensor_adjs)
        return frame_expansion_adjs

    @functools.lru_cache(maxsize=10)
    def calculate_frame_tensor_adjs(
            self, mode='elemental', n_hop=1, order1_only=True, tensor_rank=1):
        """Calculate frame adjacency matrices F s.t. phi = F phi,
        based on frame which are set of relative vectors coming from mesh
        topology.

        Parameters
        ----------
        mode: ['nodal', 'elemental'], optional
            If 'nodal', deal with nodal data. If 'elemental', deal with
            elemental data. The default is 'nodal'.
        n_hop: int, optional
            The number of hops to be considered. The default is 1.
        order1_only: bool, optional
            If True, take into account only order 1 nodes. The default is True.
        tensor_rank: int, optional
            Tensor rank for the generated frame.

        Returns
        -------
        frame_tensor_adjs: List[scipy.sparse.csr_matrix]
            List of shape (dimension, dimension, ...) which contains
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          k repetitions for rank k tensor
            (n_node, n_node) shaped sparse matrices.
        scaled_tensor_adjs: List[scipy.sparse.csr_matrix]
            List of shape (dimension, dimension, ...) which contains
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          k*2 repetitions for rank k tensor
            (n_node, n_node) shaped sparse matrices.
        """
        if tensor_rank > 1:
            raise NotImplementedError

        if mode == 'elemental':
            positions = self.convert_nodal2elemental(
                'NODE', calc_average=True)

        elif mode == 'nodal':
            if order1_only:
                filter_ = self.filter_first_order_nodes()
            else:
                filter_ = np.ones(len(self.nodes.ids), dtype=bool)
            positions = self.nodal_data.get_attribute_data('NODE')[filter_]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        adj = self.calculate_n_hop_adj(
            mode=mode, n_hop=n_hop, include_self_loop=False,
            order1_only=order1_only)

        diff_position_adjs = self.calculate_data_diff_adjs(adj, positions)

        def sum_axis_1(x):
            return np.array(np.sum(x, axis=1))[:, 0]
        moment_tensors = np.transpose(self._operate_sparse_list(
            self.calculate_tensor_power(diff_position_adjs, power=2),
            sum_axis_1), [-1, 0, 1])
        inversed_moment_tensors = self._inverse_tensors(moment_tensors)

        scaled_diff_position_adjs = self._dot_ndarray_sparse(
            inversed_moment_tensors, diff_position_adjs)

        frame_tensor_adjs = self.multiply_sparse_tensors(
            scaled_diff_position_adjs, diff_position_adjs)
        powered_frame_tensor_adjs = self.calculate_tensor_power(
            frame_tensor_adjs, power=tensor_rank)

        return powered_frame_tensor_adjs

    def _calculate_inner_product_adj(self, array, sparses):
        n = array.shape[0]
        sparse_tensor_rank = len(array.shape) - 1
        dim = array.shape[-1]

        sparse_component = sp.csr_matrix((n, n))
        if sparse_tensor_rank == 1:
            for i_array_col in range(dim):
                sparse_component = sparse_component + sparses[
                    i_array_col].multiply(array[:, i_array_col, None])
        elif sparse_tensor_rank == 2:
            for i_array_row in range(dim):
                for i_array_col in range(dim):
                    sparse_component = sparse_component + sparses[
                        i_array_row][i_array_col].multiply(
                            array[:, i_array_row, i_array_col, None])
        else:
            raise NotImplementedError
        return sparse_component

    def _dot_ndarray_sparse(self, array, sparses):
        frame_tensor_rank = len(array.shape) - 2
        dim = array.shape[-1]

        if frame_tensor_rank == 1:
            list_sparses = []
            for i_array_row in range(dim):
                sparse_component = self._calculate_inner_product_adj(
                    array[:, i_array_row], sparses)
                list_sparses.append(sparse_component)
        else:
            raise NotImplementedError

        return list_sparses

    def _inverse_tensors(self, tensors):
        tensor_rank = len(tensors.shape) - 1
        if tensor_rank == 2:
            return np.linalg.inv(tensors)
        elif tensor_rank == 4:
            # Calculate inverse of rank 4 tensors based on:
            # https://math.stackexchange.com/a/1625168/826161
            # https://www.physicsforums.com/threads/inverse-of-a-4th-order-tensor.806227/  # NOQA
            n = tensors.shape[0]
            dim = tensors.shape[-1]
            flattened_tensors = np.reshape(tensors, (n, dim**2, dim**2))
            inversed = np.linalg.inv(flattened_tensors)
            return np.reshape(inversed, (n, dim, dim, dim, dim))
        else:
            raise NotImplementedError

    def calculate_tensor_power(self, x, power, is_sparse=None):
        """Calculate tensor power of x.

        Parameters:
            x: numpy.ndarray or List[scipy.sparse]
                In case of dense, R^{n * d^k}, in case of sparse,
                R^{d^k * n * n}, where n is the number of node, k is the rank.
            power: int

        Returns:
            tensor_power_x: numpy.ndarray or List[scipy.sparse]
        """
        if power == 1:
            return x
        if power < 1:
            raise ValueError(f"Invalid tensor power: {power}")

        if is_sparse is None:
            if isinstance(x, list):
                is_sparse = True

            else:
                is_sparse = False

        if not is_sparse:
            tensor_rank = len(x.shape) - 1

            if tensor_rank > 6:
                raise NotImplementedError
            if tensor_rank < 1:
                raise ValueError(f"Invalid input data shape: {x.shape}")
            if tensor_rank * power > 18:
                raise NotImplementedError

        smaller_powered = self.calculate_tensor_power(
            x, power-1, is_sparse=is_sparse)

        if is_sparse:
            tensor_powered = self.multiply_sparse_tensors(smaller_powered, x)

        else:
            smaller_powered_rank = tensor_rank * (power - 1)
            einstring_1 = 'abcdefghijklmnopqrs'[:1+smaller_powered_rank]
            einstring_2 = 'atuvwxy'[:1+tensor_rank]
            tensor_powered = np.einsum(
                einstring_1 + ',' + einstring_2
                + '->' + einstring_1 + einstring_2[1:],
                smaller_powered, x)

        return tensor_powered

    def multiply_sparse_tensors(self, a, b):
        """Perform multiplication of sparse tensors.

        Parameters
        ----------
        a: List[scipy.sparse] or scipy.sparse
        b: List[scipy.sparse] or scipy.sparse

        Returns
        -------
        c: List[scipy.sparse] or scipy.sparse
        """
        if isinstance(b, (sp.coo_matrix, sp.csr_matrix)):
            if isinstance(a, (sp.coo_matrix, sp.csr_matrix)):
                return a.multiply(b)
            else:
                return self.multiply_sparse_tensors(b, a)  # Strip a in tern
        return [self.multiply_sparse_tensors(a, _b) for _b in b]

    def _get_original_sparses(self, sparse_list):
        if isinstance(sparse_list, (sp.coo_matrix, sp.csr_matrix)):
            return [sparse_list]
        ret = []
        for s in sparse_list:
            ret = ret + self._get_original_sparses(s)
        return s

    def _operate_sparse_list(self, sparse_list, callback):
        if isinstance(sparse_list, (sp.coo_matrix, sp.csr_matrix)):
            return callback(sparse_list)
        return [
            self._operate_sparse_list(s, callback=callback)
            for s in sparse_list]

    def _sparsify_tensor(self, data_array, original_sparse):
        if len(data_array.shape) == 1:
            if isinstance(original_sparse, sp.coo_matrix):
                return sp.coo_matrix(
                    (data_array, (original_sparse.row, original_sparse.col)),
                    shape=original_sparse.shape)
            elif isinstance(original_sparse, sp.csr_matrix):
                indices = original_sparse.indices
                indptr = original_sparse.indptr
                return sp.csr_matrix(
                    (data_array, indices, indptr), shape=original_sparse.shape)
        else:
            dim = data_array.shape[-1]
            return [
                self._sparsify_tensor(data_array[:, i], original_sparse)
                for i in range(dim)]

    def calculate_elemental_spatial_gradients(
            self, elemental_data, n_hop=1, kernel=None, **kwargs):
        """Calculate spatial gradient (not graph gradient) w.r.t elemental
        data.

        Parameters
        ----------
        elemental_data : numpy.ndarray
            Data to calculate gradient over. It should be
            (n_element, n_feature)-shaped array.

        Returns
        -------
        gradients: numpy.ndarray
            (n_element, 3, n_feature)-shaped array of grad. 3 means dimension
            of the space.
        """
        grad_adjs = self.calculate_spatial_gradient_adjacency_matrices(
            mode='elemental', n_hop=n_hop, kernel=kernel, **kwargs)
        return np.stack([
            grad_adj.dot(elemental_data) for grad_adj in grad_adjs], axis=1)

    def calculate_nodal_spatial_gradients(
            self, nodal_data, n_hop=1, kernel=None, order1_only=True,
            **kwargs):
        """Calculate spatial gradient (not graph gradient) w.r.t nodal
        data.

        Parameters
        ----------
        nodal_data: numpy.ndarray
            Data to calculate gradient over. It should be
            (n_node, n_feature)-shaped array.
        n_hop: int, optional [1]
            The number of hops to be considered.
        kernel: str, optional [None]
            Kernel function type.
        order1_only: bool, optional [True]
            If True, consider only order 1 nodes.

        Returns
        -------
        gradients: numpy.ndarray
            (n_node, 3, n_feature)-shaped array of grad. 3 means dimension
            of the space.
        """
        grad_adjs = self.calculate_spatial_gradient_adjacency_matrices(
            mode='nodal', n_hop=n_hop, kernel=kernel, order1_only=order1_only,
            **kwargs)
        if order1_only:
            filter_ = self.filter_first_order_nodes()
        else:
            filter_ = np.ones(len(self.nodes.ids), dtype=bool)
        return np.stack([
            grad_adj.dot(nodal_data[filter_])
            for grad_adj in grad_adjs], axis=1)

    def calculate_spatial_gradient_adjacency_matrices(
            self, mode='elemental', n_hop=1, kernel=None, order1_only=True,
            use_effective_volume=True, moment_matrix=False,
            consider_volume=True, **kwargs):
        """Calculate spatial gradient (not graph gradient) matrix.

        Parameters
        ----------
        nodal_data: numpy.ndarray
            Data to calculate gradient over. It should be
            (n_node, n_feature)-shaped array.
        n_hop: int, optional [1]
            The number of hops to be considered.
        kernel: str, optional [None]
            Kernel function type.
        order1_only: bool, optional [True]
            If True, consider only order 1 nodes.
        moment_matrix: bool, optional [False]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [True]
            If True, consider effective volume of each vertex.

        Returns
        -------
        gradients: numpy.ndarray
            (n_node, 3, n_feature)-shaped array of grad. 3 means dimension
            of the space.
        """
        if mode == 'elemental':
            positions = self.convert_nodal2elemental(
                'NODE', calc_average=True)
            if consider_volume:
                volumes = self.calculate_element_volumes()

        elif mode == 'nodal':
            if order1_only:
                filter_ = self.filter_first_order_nodes()
            else:
                filter_ = np.ones(len(self.nodes.ids), dtype=bool)
            positions = self.nodal_data.get_attribute_data('NODE')[filter_]

            if consider_volume:
                if use_effective_volume:
                    volumes = self.convert_elemental2nodal(
                        self.calculate_element_volumes(), mode='effective',
                        order1_only=order1_only)
                else:
                    volumes = self.convert_elemental2nodal(
                        self.calculate_element_volumes(), mode='mean',
                        order1_only=order1_only)[filter_]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        adj = self.calculate_n_hop_adj(
            mode=mode, n_hop=n_hop, include_self_loop=False,
            order1_only=order1_only)

        diff_position_adjs = self.calculate_data_diff_adjs(adj, positions)
        distance_adj = self.calculate_norm_adj(diff_position_adjs)

        if consider_volume:
            volume_adj = self.calculate_data_adjs(adj, volumes)[0]
        else:
            volume_adj = adj.astype(float)

        if kernel is None:
            weight_adj = volume_adj
        else:
            weight_adj = self.calculate_distance_kernel_adj(
                kernel, distance_adj, **kwargs
            ).multiply(volume_adj)

        if moment_matrix:
            weight_by_squarenorm_adj = distance_adj.power(-2).multiply(
                weight_adj)
            # normalized_diff_position_adjs = [
            #     distance_adj.power(-1).multiply(diff_position_adj)
            #     for diff_position_adj in diff_position_adjs]

            def sum_axis_1_with_weight(x):
                return np.array(np.sum(x.multiply(
                    weight_by_squarenorm_adj), axis=1))[:, 0]
            moment_tensors = np.transpose(self._operate_sparse_list(
                self.calculate_tensor_power(
                    diff_position_adjs, power=2),
                sum_axis_1_with_weight), [-1, 0, 1])
            inversed_moment_tensors = self._inverse_tensors(moment_tensors)
            grad_adj_wo_selfs = self._dot_ndarray_sparse(
                inversed_moment_tensors, [
                    diff_position_adj.multiply(weight_by_squarenorm_adj)
                    for diff_position_adj in diff_position_adjs])

        else:
            summed_weight = np.array(weight_adj.sum(axis=1))
            dim = 3  # NOTE: assume always 3d simulation

            #  x_j - x_i         w_{ij}
            # ------------ ------------------
            #    d_{ij}^2   w_{ik} delta_{kk}
            grad_adj_wo_selfs = [
                dim * distance_adj.power(-2).multiply(
                    diff_position_adj).multiply(
                        weight_adj).multiply(summed_weight**-1)
                for diff_position_adj in diff_position_adjs]
        grad_adjs = [
            sp.coo_matrix(
                grad_adj_wo_self - sp.eye(*grad_adj_wo_self.shape).multiply(
                    grad_adj_wo_self.sum(axis=1)))
            for grad_adj_wo_self in grad_adj_wo_selfs]
        return grad_adjs

    def calculate_distance_kernel_adj(
            self, kernel, distance_adj, **kwargs):
        if kernel == 'exp':
            data = self.calculate_distance_kernel_adj_exp_data(
                distance_adj.data, **kwargs)
        elif kernel == 'gauss':
            data = self.calculate_distance_kernel_adj_gauss_data(
                distance_adj.data, **kwargs)
        elif kernel == 'chisquare':
            data = self.calculate_distance_kernel_adj_chisquare_data(
                distance_adj.data, **kwargs)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        return sp.csr_matrix((data, distance_adj.indices, distance_adj.indptr))

    def calculate_distance_kernel_adj_exp_data(self, adj_data, alpha=1.):
        return np.exp(- alpha * adj_data)

    def calculate_distance_kernel_adj_gauss_data(self, adj_data, alpha=1.):
        return np.exp(- .5 * alpha * adj_data**2.)

    def calculate_distance_kernel_adj_chisquare_data(self, adj_data, k=1):
        return adj_data**(k/2 - 1) * np.exp(- .5 * adj_data)

    def calculate_norm_adj(self, adjs, p=2):
        return np.sum([adj.power(p) for adj in adjs]).power(1 / p)

    def calculate_data_adjs(self, adj, data):
        return [adj.multiply(d) for d in data.T]

    def calculate_data_diff_adjs(self, adj, data):
        # NOTE: Non zero profile could be different for each component
        data_adjs = self.calculate_data_adjs(adj, data)
        return [data_adj - data_adj.T for data_adj in data_adjs]

    def calculate_edge_differences(
            self, data, *, mode='elemental', include_self_loop=False):
        """Calculate difference between data which are connected with edges
        (in terms of either elemental or nodal graph).

        Parameters
        ----------
        data : numpy.ndarray
            data to be subtracted. It shoud be (n_vertex, 2, n_feature)-shaped
            array which [:, 0, :] contains reference vertex and [:, 1, :]
            contains opposed vertex.
        mode : str, optional
            Mode of graph, which is either 'elemental' or 'nodal'. Defaults to
            'elemental'.
        include_self_loop : bool, optional
            If True, include self loop to the operation. Defaults to False.

        Returns
        -------
        differentiated_data: numpy.ndarray
            (n_vertex, n_feature)-shaped array after subtraction.
        """
        edge_data = self.convert_vertices_to_edges(
            data, mode=mode, include_self_loop=include_self_loop)
        diff_data = edge_data[:, 1] - edge_data[:, 0]
        return diff_data

    def convert_vertices_to_edges(
            self, data, mode='elemental', include_self_loop=False):
        """Convert vertex values to edge values (in terms of either elemental
        or nodal graph).

        Parameters
        ----------
        data : numpy.ndarray
            Vertex data, which should be (n_vertex, n_feature)-shaped.
        mode : str, optional
            Mode of graph, which is either 'elemental' or 'nodal'. Defaults to
            'elemental'.
        include_self_loop : bool, optional
            If True, include self loop to the operation. Defaults to False.

        Returns
        -------
        edge_values: numpy.ndarray
            (n_element, 2, n_feature)-shaped array of edge data. [:, 0, :]
            contains reference vertex and [:, 1, :] contains opposed vertex.
        """
        if mode == 'elemental':
            adj = self.calculate_adjacency_matrix_element()
        elif mode == 'nodal':
            adj = self.calculate_adjacency_matrix_node()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if not include_self_loop:
            adj = sp.coo_matrix(adj - sp.eye(*adj.shape))

        edge_data = np.stack([data[adj.row], data[adj.col]], axis=1)
        return edge_data

    def aggregate_edges_to_vertices(
            self, edges_data, mode='elemental', include_self_loop=False):
        """Aggregate edge data to vertices (in terms of either elemental
        of nodal graph).

        Parameters
        ----------
        edges_data : numpy.ndarray
            Edge data, which should be either (n_edge, n_feature) or
            (n_edge, n_feature1, n_feature2) -shaped array.
        mode : str, optional
            Mode of graph, which is either 'elemental' or 'nodal'. Defaults to
            'elemental'.
        include_self_loop : bool, optional
            If True, include self loop to the operation. Defaults to False.

        Returns
        -------
        aggregated_vertice_data: numpy.ndarray
            Either (n_vertex, n_feature) or (n_vertes, n_feature1, n_feature2)
            -shaped array of aggregated vertex data.
        """
        e2v_matrix = self.calculate_e2v_matrix(
            mode=mode, include_self_loop=include_self_loop)
        if len(edges_data.shape) == 2:
            return e2v_matrix @ edges_data
        elif len(edges_data.shape) == 2:
            return np.stack([
                e2v_matrix @ d
                for d in np.transpose(edges_data, (2, 0, 1))], axis=-1)
        elif len(edges_data.shape) == 3:
            return np.stack([
                e2v_matrix @ d
                for d in np.transpose(edges_data, (2, 0, 1))], axis=-1)
        else:
            raise ValueError(f"Unsupported data shape: {edges_data.shape}")

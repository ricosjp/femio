import femio
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from functools import lru_cache
import datetime


class MeshCompressor:
    def __init__(self, *, fem_data,
                 ):
        self.fem_data = fem_data
        self.csr_raw = self.fem_data.face_data_csr()
        self.csr = self.csr_raw
        self.node_pos_raw = fem_data.nodes.data.copy()
        self.node_pos = fem_data.nodes.data.copy()
        self.elem_conv = np.arange(
            len(fem_data.elements.data), dtype=np.int32)
        self.node_conv = np.arange(
            len(fem_data.nodes.data), dtype=np.int32)
        self.done = 0

    def compress(self, *,
                 elem_num, cos_thresh, dist_thresh
                 ):
        """
        Compress the input FEMData.
        After running this method, you can calculate the compressed FEMData
        by calculate_compressed_fem_data method.

        Args:
            elem_num: int
                The number of elements of output FEMData.
                This param is just an estimate.
                Typically, the number of output FEMData will be
                1.0x ~ 2.0x, where x = elem_num.
            cos_thresh:
                Two faces, edges are merged if cos of its angle is
                greater than cos_thresh.
                Therefore, if cos_thresh is near 1.0 then the overall shape
                of the FEMData tends to be preserved.
            dist_thresh:
                The nodes connected by a edge is merged if distance between
                them is shorter than dist_thresh.
                Therefore, if dist_thresh is near 0.0 then the overall shape
                of the FEMData tends to be preserved.
        """
        assert self.done == 0
        self.done = 1
        print(datetime.datetime.now())
        print_stat(self.csr)
        print("merge elements")
        K = len(self.csr[0] - 1) // elem_num
        K = max(K, 1)
        self.csr = merge_elements(
            self.csr, self.node_pos, self.elem_conv, K)
        for _ in range(10):
            before_size = len(self.csr[1])
            print(datetime.datetime.now())
            print_stat(self.csr)
            print("remove edges")
            self.csr = remove_edges(
                self.csr,
                self.node_pos,
                self.elem_conv,
                THRESH=cos_thresh)
            print(datetime.datetime.now())
            print("remove vertices")
            self.csr = remove_vertices_2(
                self.csr, self.node_pos, self.elem_conv)
            print(datetime.datetime.now())
            print("merge vertices")
            self.csr, self.node_pos = merge_vertices(
                self.csr, self.node_pos, self.elem_conv, self.node_conv,
                THRESH=dist_thresh)
            if before_size == len(self.csr[1]):
                break
        reindex(self.csr, self.node_conv)
        self.node_pos = recalc_node_pos(self.node_pos_raw, self.node_conv)
        if len(self.node_pos) == 0:
            print("compressed to 0 elements. try another parameter")
            return False
        self.calculate_compressed_fem_data()
        return True

    @lru_cache
    def calculate_compressed_fem_data(self):
        """
        Calculate the compressed FEMData.
        Before running this method, you need to run compress method.

        Returns:
            output_fem_data: FEMData Object
        """
        assert self.done
        node_pos = self.node_pos
        indptr, faces = self.csr
        P = len(indptr) - 1
        nodes = femio.FEMAttribute(
            'nodes',
            ids=np.arange(len(node_pos)) + 1,
            data=node_pos)
        element_data = np.empty(P, object)
        for p in range(P):
            element_data[p] = collect_vertex(
                faces[indptr[p]:indptr[p + 1]]) + 1
        polyhedron = femio.FEMAttribute(
            'polyhedron',
            ids=np.arange(len(element_data)) + 1,
            data=element_data)
        elements = femio.FEMElementalAttribute(
            'ELEMENT', {'polyhedron': polyhedron}
        )
        fem_data = femio.FEMData(nodes=nodes, elements=elements)
        face_data_list = np.empty(P, object)  # ndarray of list
        for p in range(P):
            face_data_list[p] = list(faces[indptr[p]:indptr[p + 1]])
        fem_data.elemental_data.update(
            {
                'face': femio.FEMElementalAttribute(
                    'face', {
                        'polyhedron': femio.FEMAttribute(
                            'face', ids=np.arange(P) + 1,
                            data=face_data_list)})})
        print("nodes", len(fem_data.nodes.data))
        print("elements", len(fem_data.elements.data))
        self.output_fem_data = fem_data
        return self.output_fem_data

    @lru_cache
    def calculate_conversion_matrix_nodal(self, knn):
        """
        Calculate the conversion matrix of nodal_data.
        Let N be the number of the original FEMData, and
        M be the number of the original FEMData.
        Then create a sparse (M, N) matrix.

        We can convert nodal_data between original FEMData and compressed
        FEMData by using this matrix.
        We can also convert them by using compress_nodal_data and
        decompress_nodal_data method.

        Args:
            knn: int
                The number of nodes related to a node in original fem_data.
                In other words, the number of nonzero entry in each columns
                in the calculated matrix.
        Returns:
            mat: csr_matrix (bool)
                The conversion matrix of nodal_data.
                Its shape is (M, N).
        """
        assert self.done
        nbd = calculate_nodal_knn(
            self.csr_raw, self.node_conv, self.node_pos, knn)
        N = len(self.node_conv)
        M = self.node_conv.max() + 1
        rows = np.empty(knn * N, np.int32)
        for k in range(knn):
            rows[k::knn] = np.arange(N)
        cols = nbd.ravel()
        idx = cols != -1
        rows = rows[idx]
        cols = cols[idx]
        vals = np.ones(len(rows), np.bool_)
        mat = csr_matrix((vals, (cols, rows)), (M, N), dtype=np.bool_)
        return mat

    def compress_nodal_data(self, *, name_1, name_2, kind, knn):
        """
        Convert nodal_data in original FEMData to compressed FEMData.
        Converted nodal_data is automatically attached to the
        compressed FEMData.

        Args:
            name_1: str
                Name of the nodal_data in original FEMData.
            name_2: str
                Name of the nodal_data in compressed FEMData
                which will be created.
            kind: str, "sum" or "mean"
                If kind == "mean", new data is computed as simple average of
                original data related to it.
                If kind == "sum", new data is computed as weighted sum of
                original data related to it, and the sum of data of all nodes
                is preserved.
            knn: int
                The number of nodes related to a node in
                original fem_data.
        """
        if kind not in ["sum", "mean"]:
            raise ValueError
        mat = self.calculate_conversion_matrix_nodal(knn)
        x = self.fem_data.nodal_data[name_1].data
        if kind == "mean":
            y = mat @ x
            wt = mat.sum(axis=1)
            y = y / wt
            ids = self.output_fem_data.nodes.ids
            self.output_fem_data.nodal_data.update_data(
                ids, {
                    name_2: y
                }
            )
        if kind == "sum":
            wt = mat.sum(axis=0)
            x = x / wt
            y = mat @ x
            ids = self.output_fem_data.nodes.ids
            self.output_fem_data.nodal_data.update_data(
                ids, {
                    name_2: y
                }
            )

    def decompress_nodal_data(self, *, name_1, name_2, kind, knn):
        """
        Convert nodal_data in compressed FEMData to original FEMData.
        Converted nodal_data is automatically attached to the
        original FEMData.

        Args:
            name_1: str
                Name of the nodal_data in compressed FEMData.
            name_2: str
                Name of the nodal_data in original FEMData
                which will be created.
            kind: str, "sum" or "mean"
                If kind == "mean", new data is computed as simple average of
                original data related to it.
                If kind == "sum", new data is computed as weighted sum of
                original data related to it, and the sum of data of all nodes
                is preserved.
            knn: int
                The number of nodes related to a node in
                original fem_data.
        """
        if kind not in ["sum", "mean"]:
            raise ValueError
        mat = self.calculate_conversion_matrix_nodal(knn)
        mat = mat.T
        x = self.output_fem_data.nodal_data[name_1].data
        if kind == "mean":
            y = mat @ x
            wt = mat.sum(axis=1)
            y = y / wt
            ids = self.fem_data.nodes.ids
            self.fem_data.nodal_data.update_data(
                ids, {
                    name_2: y
                }
            )
        if kind == "sum":
            wt = mat.sum(axis=0)
            x = x / wt
            y = mat @ x
            ids = self.fem_data.nodes.ids
            self.fem_data.nodal_data.update_data(
                ids, {
                    name_2: y
                }
            )

    @lru_cache
    def calculate_conversion_matrix_elemental(self, knn):
        """
        Calculate the conversion matrix of element_data.
        Let N be the number of the original FEMData, and
        M be the number of the original FEMData.
        Then create a sparse (M, N) matrix.

        We can convert elemental_data between original FEMData and compressed
        FEMData by using this matrix.
        We can also convert them by using compress_elemental_data_data and
        decompress_elemental_data_data method.

        Args:
            knn: int
                The number of elements related to a element in
                original fem_data.
                In other words, the number of nonzero entry in each columns
                in the calculated matrix.
        Returns:
            mat: csr_matrix (bool)
                The conversion matrix of elemental_data.
                Its shape is (M, N).
        """
        assert self.done
        N = len(self.csr_raw[0]) - 1
        M = len(self.csr[0]) - 1
        nbd1, nbd2 = calculate_elemental_knn(
            self.csr_raw, self.csr, self.node_conv, self.node_pos, knn)
        rows1 = np.empty(knn * N, np.int32)
        for k in range(knn):
            rows1[k::knn] = np.arange(N)
        cols1 = nbd1.ravel()
        cols2 = np.empty(knn * M, np.int32)
        for k in range(knn):
            cols2[k::knn] = np.arange(M)
        rows2 = nbd2.ravel()
        rows = np.concatenate([rows1, rows2])
        cols = np.concatenate([cols1, cols2])
        idx = (rows != -1) & (cols != -1)
        rows = rows[idx]
        cols = cols[idx]
        vals = np.ones(len(rows), np.bool_)
        mat = csr_matrix((vals, (cols, rows)), (M, N), dtype=np.bool_)
        return mat

    def compress_elemental_data(self, *, name_1, name_2, kind, knn):
        """
        Convert elemental_data in original FEMData to compressed FEMData.
        Converted elemental_data is automatically attached to the
        compressed FEMData.

        Args:
            name_1: str
                Name of the elemental_data in original FEMData.
            name_2: str
                Name of the elemental_data in compressed FEMData
                which will be created.
            kind: str, "sum" or "mean"
                If kind == "mean", new data is computed as simple average of
                original data related to it.
                If kind == "sum", new data is computed as weighted sum of
                original data related to it, and the sum of data of
                all elements is preserved.
            knn: int
                The number of elements related to a element in
                original fem_data.
        """
        if kind not in ["sum", "mean"]:
            raise ValueError
        mat = self.calculate_conversion_matrix_elemental(knn)
        x = self.fem_data.elemental_data[name_1].data
        if kind == "mean":
            y = mat @ x
            wt = mat.sum(axis=1)
            y = y / wt
            ids = self.output_fem_data.elements.ids
            self.output_fem_data.elemental_data.update_data(
                ids, {
                    name_2: y
                }
            )
        if kind == "sum":
            wt = mat.sum(axis=0)
            x = x / wt
            y = mat @ x
            ids = self.output_fem_data.elements.ids
            self.output_fem_data.elemental_data.update_data(
                ids, {
                    name_2: y
                }
            )

    def decompress_elemental_data(self, *, name_1, name_2, kind, knn):
        """
        Convert elemental_data in compressed FEMData to original FEMData.
        Converted elemental_data is automatically attached to the
        original FEMData.

        Args:
            name_1: str
                Name of the elemental_data in compressed FEMData.
            name_2: str
                Name of the elemental_data in original FEMData
                which will be created.
            kind: str, "sum" or "mean"
                If kind == "mean", new data is computed as simple average of
                original data related to it.
                If kind == "sum", new data is computed as weighted sum of
                original data related to it, and the sum of data of
                all elements is preserved.
            knn: int
                The number of elements related to a element in
                original fem_data.
        """
        if kind not in ["sum", "mean"]:
            raise ValueError
        mat = self.calculate_conversion_matrix_elemental(knn)
        mat = mat.T
        x = self.output_fem_data.elemental_data[name_1].data
        if kind == "mean":
            y = mat @ x
            wt = mat.sum(axis=1)
            y = y / wt
            ids = self.fem_data.elements.ids
            self.fem_data.elemental_data.update_data(
                ids, {
                    name_2: y
                }
            )
        if kind == "sum":
            wt = mat.sum(axis=0)
            x = x / wt
            y = mat @ x
            ids = self.fem_data.elements.ids
            self.fem_data.elemental_data.update_data(
                ids, {
                    name_2: y
                }
            )


@njit
def calculate_nodal_knn(csr, node_conv, node_pos, knn):
    N = len(node_conv)
    G_li = [(0, 0)] * 0
    indptr, face_data = csr
    P = len(indptr) - 1
    for p in range(P):
        V = collect_vertex(face_data[indptr[p]:indptr[p + 1]])
        for v in V:
            G_li.append((v, p))
    G_ve = np.array(G_li)
    G_ev = G_ve[:, ::-1]
    G_ve = G_ve[np.argsort(G_ve[:, 0])]
    indptr_ve = np.zeros(N + 1, np.int32)
    indptr_ev = np.zeros(P + 1, np.int32)
    for i in range(len(G_ve)):
        v, e = G_ve[i]
        indptr_ve[v + 1] += 1
        indptr_ev[e + 1] += 1
    for v in range(N):
        indptr_ve[v + 1] += indptr_ve[v]
    for e in range(P):
        indptr_ev[e + 1] += indptr_ev[e]
    nbd = np.full((N, knn), -1, np.int32)
    que = np.empty((knn * N, 2), np.int32)
    ql = qr = 0
    for v in range(N):
        if node_conv[v] != -1:
            nbd[v, 0] = v
            que[qr] = (v, v)
            qr += 1
    while ql < qr:
        v, frm = que[ql]
        a = ql * 10 // len(que)
        ql += 1
        b = ql * 10 // len(que)
        if a < b:
            print(ql, "/", len(que))
        for i in range(indptr_ve[v], indptr_ve[v + 1]):
            assert G_ve[i, 0] == v
            e = G_ve[i, 1]
            for j in range(indptr_ev[e], indptr_ev[e + 1]):
                assert G_ev[j, 0] == e
                to = G_ev[j, 1]
                if np.any(nbd[to] == frm):
                    continue
                for k in range(knn):
                    if nbd[to, k] == -1:
                        nbd[to, k] = frm
                        que[qr, 0] = to
                        que[qr, 1] = frm
                        qr += 1
                        break
    for v in range(N):
        for k in range(knn):
            if nbd[v][k] == -1:
                continue
            nbd[v][k] = node_conv[nbd[v][k]]
    W = np.where(node_conv != -1)[0]
    V = np.where(nbd[:, 0] == -1)
    for i in range(len(V)):
        v = V[i]
        min_dist = np.inf
        for w in W:
            d = ((node_pos[v] - node_pos[w]) ** 2).sum()
            if min_dist > d:
                min_dist = d
                nbd[v][0] = node_conv[w]
    return nbd


@njit
def calculate_elemental_knn(csr_raw, csr_after, node_conv, node_pos, knn):
    nbd = calculate_nodal_knn(csr_raw, node_conv, node_pos, knn)

    def calc_from_raw():
        indptr, face_data = csr_after
        P = len(indptr) - 1
        G_li = [(0, 0)] * 0
        for p in range(P):
            V = collect_vertex(face_data[indptr[p]:indptr[p + 1]])
            for v in V:
                G_li.append((v, p))
        G_ve = np.array(G_li)
        G_ve = G_ve[np.argsort(G_ve[:, 0])]
        N = G_ve[:, 0].max() + 1
        indptr_ve = np.searchsorted(G_ve[:, 0], np.arange(N + 1))

        Q = len(csr_raw[0]) - 1
        res = np.full((Q, knn), -1, np.int32)
        indptr, face_data = csr_raw
        for q in range(Q):
            poly = face_data[indptr[q]:indptr[q + 1]]
            V = collect_vertex(poly)
            E = [0] * 0
            for v in V:
                for w in nbd[v]:
                    for e in G_ve[indptr_ve[w]:indptr_ve[w + 1], 1]:
                        E.append(e)
            key = np.unique(np.array(E))
            K = len(key)
            cnt = np.zeros(K, np.int32)
            for e in E:
                idx = np.searchsorted(key, e)
                cnt[idx] += 1
            IDS = np.argsort(cnt)
            IDS = IDS[::-1][:knn]
            for k in range(len(IDS)):
                res[q][k] = key[IDS[k]]
        return res

    def calc_from_after():
        indptr, face_data = csr_raw
        P = len(indptr) - 1
        G_li = [(0, 0)] * 0
        for p in range(P):
            V = collect_vertex(face_data[indptr[p]:indptr[p + 1]])
            for v in V:
                G_li.append((node_conv[v], p))
        G_ve = np.array(G_li)
        G_ve = G_ve[np.argsort(G_ve[:, 0])]
        N = G_ve[:, 0].max() + 1
        indptr_ve = np.searchsorted(G_ve[:, 0], np.arange(N + 1))

        Q = len(csr_after[0]) - 1
        res = np.full((Q, knn), -1, np.int32)
        indptr, face_data = csr_after
        for q in range(Q):
            poly = face_data[indptr[q]:indptr[q + 1]]
            V = collect_vertex(poly)
            E = [0] * 0
            for v in V:
                for w in nbd[v]:
                    for e in G_ve[indptr_ve[w]:indptr_ve[w + 1], 1]:
                        E.append(e)
            key = np.unique(np.array(E))
            K = len(key)
            cnt = np.zeros(K, np.int32)
            for e in E:
                idx = np.searchsorted(key, e)
                cnt[idx] += 1
            IDS = np.argsort(cnt)
            IDS = IDS[::-1][:knn]
            for k in range(len(IDS)):
                res[q][k] = key[IDS[k]]
        return res
    return calc_from_raw(), calc_from_after()


@njit
def recalc_node_pos(node_pos, node_conv):
    N = len(node_pos)
    K = node_conv.max() + 1
    res = np.zeros((K, 3), np.float64)
    cnt = np.zeros(K, np.int32)
    for v in range(N):
        k = node_conv[v]
        if k == -1:
            continue
        cnt[k] += 1
        res[k] += node_pos[v]
    for k in range(K):
        res[k] /= cnt[k]
    return res


@njit
def reindex(csr, node_conv):
    indptr, dat = csr
    N = len(node_conv)
    P = len(indptr) - 1
    isin = np.zeros(N, np.bool_)
    for p in range(P):
        poly = dat[indptr[p]:indptr[p + 1]]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for v in F:
                isin[v] = 1

    while True:
        ok = 1
        for v in range(N):
            p = node_conv[v]
            assert p != -1
            if node_conv[p] == p:
                continue
            ok = 0
            node_conv[v] = node_conv[p]
        if ok:
            break

    for v in range(N):
        p = node_conv[v]
        if not isin[p]:
            node_conv[v] = -1

    new_ids = np.empty(N, np.int32)
    nxt_idx = 0
    for v in range(N):
        if not isin[v]:
            new_ids[v] = -1
            continue
        new_ids[v] = nxt_idx
        nxt_idx += 1

    for v in range(N):
        if node_conv[v] == -1:
            continue
        p = node_conv[v]
        node_conv[v] = new_ids[p]

    for p in range(P):
        poly = dat[indptr[p]:indptr[p + 1]]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for i in range(len(F)):
                v = F[i]
                F[i] = new_ids[v]
    return


@njit
def shrink(polyhedrons, elem_conv):
    P = len(polyhedrons)
    new_ids = np.full(P, -1, np.int32)
    nxt_idx = 0
    for p in range(P):
        poly = polyhedrons[p]
        m = poly[0]
        poly[0] = 0
        i = 1
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            if k >= 3:
                poly[i] = k
                poly[i + 1:i + k + 1] = F
                i += k + 1
                poly[0] += 1
        poly = poly[:L]
        polyhedrons[p] = poly
        if poly[0] <= 2:
            continue
        new_ids[p] = nxt_idx
        nxt_idx += 1
    for i in range(len(elem_conv)):
        e = elem_conv[i]
        if e == -1:
            continue
        elem_conv[i] = new_ids[e]
    polyhedrons = [poly for poly in polyhedrons if poly[0] > 2]
    return polyhedrons, elem_conv


@njit
def remove_one_vertex_from_polyhedron(poly, rm_v):
    # rm_v を含む face をすべてマージしたい。
    m = poly[0]
    contain = np.zeros(m, np.bool_)
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        for i in range(len(F)):
            if F[i] == rm_v:
                contain[f] = 1
    assert L == len(poly)
    if np.sum(contain) == 0:
        return False, poly
    v_list = [0] * 0
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        if not contain[f]:
            continue
        for v in F:
            if v != rm_v:
                v_list.append(v)
    assert L == len(poly)
    V = np.unique(np.array(v_list))
    nxt = np.full(len(V), -1, np.int32)
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        if not contain[f]:
            continue
        for i in range(len(F)):
            a = F[i - 1]
            b = F[i]
            if a == rm_v or b == rm_v:
                continue
            ia = np.searchsorted(V, a)
            ib = np.searchsorted(V, b)
            assert V[ia] == a
            assert V[ib] == b
            if nxt[ia] != -1:
                return False, poly
            nxt[ia] = ib
    assert L == len(poly)
    cyc = np.zeros(len(V), np.int32)
    for i in range(len(V) - 1):
        cyc[i + 1] = nxt[cyc[i]]
    if len(cyc) != len(np.unique(cyc)):
        return False, poly
    cyc = V[cyc]
    newpoly = [0]
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        if contain[f]:
            continue
        if len(F) <= 2:
            continue
        newpoly[0] += 1
        newpoly.append(len(F))
        for v in F:
            newpoly.append(v)
    assert L == len(poly)
    newpoly[0] += 1
    newpoly.append(len(cyc))
    for v in cyc:
        newpoly.append(v)
    newpoly_arr = np.array(newpoly, poly.dtype)
    return True, newpoly_arr


@njit
def remove_one_edge_from_polyhedron(poly, A, B):
    # A, B を含む面をマージできるならばマージする
    m = poly[0]
    contain = np.zeros(m, np.bool_)
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        for i in range(len(F)):
            if F[i - 1] == A and F[i] == B:
                contain[f] = 1
            if F[i - 1] == B and F[i] == A:
                contain[f] = 1
    assert L == len(poly)
    if np.sum(contain) == 0:
        return True, poly
    v_list = [0] * 0
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        if not contain[f]:
            continue
        for v in F:
            v_list.append(v)
    assert L == len(poly)
    V = np.unique(np.array(v_list))
    nxt = np.full(len(V), -1, np.int32)
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        if not contain[f]:
            continue
        for i in range(len(F)):
            a = F[i - 1]
            b = F[i]
            if (a, b) == (A, B) or (a, b) == (B, A):
                continue
            ia = np.searchsorted(V, a)
            ib = np.searchsorted(V, b)
            assert V[ia] == a
            assert V[ib] == b
            if nxt[ia] != -1:
                return False, poly
            nxt[ia] = ib
    assert L == len(poly)
    cyc = np.zeros(len(V), np.int32)
    for i in range(len(V) - 1):
        cyc[i + 1] = nxt[cyc[i]]
    cyc = V[cyc]
    if len(cyc) != len(np.unique(cyc)):
        return False, poly
    newpoly = [0]
    L = 1
    for f in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        if contain[f]:
            continue
        newpoly[0] += 1
        newpoly.append(len(F))
        for v in F:
            newpoly.append(v)
    assert L == len(poly)
    newpoly[0] += 1
    newpoly.append(len(cyc))
    for v in cyc:
        newpoly.append(v)
    newpoly_arr = np.array(newpoly, poly.dtype)
    return True, newpoly_arr


@njit
def remove_edges(face_data_csr, node_pos, elem_conv, THRESH=0.99):
    indptr, dat = face_data_csr
    P = len(indptr) - 1
    polyhedrons = [dat[indptr[p]:indptr[p + 1]].copy() for p in range(P)]

    def calc_normal(F):
        vc = np.zeros(3)
        for i in range(2, len(F)):
            vc1 = node_pos[F[1]] - node_pos[F[0]]
            vc2 = node_pos[F[i]] - node_pos[F[0]]
            vc += np.cross(vc1, vc2)
        vc /= (vc * vc).sum() ** .5
        return vc

    def collect_edge(p):
        res = [0] * 0
        poly = polyhedrons[p]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                res.append(a << 32 | b)
        tmp = np.array(res, np.int64)
        return np.sort(tmp)
    # edge, pid, can remove(normal vector)
    n = 0
    for p in range(P):
        poly = polyhedrons[p]
        n += len(poly) - 1 - poly[0]
    n //= 2
    edge_data = np.empty((n, 3), np.int64)
    ptr = 0
    for p in range(P):
        # 各 edge が消せるかどうかを調べる
        edges = collect_edge(p)
        poly = polyhedrons[p]
        norms = np.full((len(edges), 3), np.nan)
        cnt = np.zeros(len(edges), np.int32)
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            norm = calc_normal(F)
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                e = a << 32 | b
                eid = np.searchsorted(edges, e)
                cnt[eid] += 1
                assert edges[eid] == e
                norms[eid] = norm
        can_rm = np.empty(len(edges), np.bool_)
        for i in range(len(edges)):
            e = edges[i]
            a, b = divmod(e, 1 << 32)
            e_rev = b << 32 | a
            j = np.searchsorted(edges, e_rev)
            assert edges[j] == e_rev
            cos_val = np.sum(norms[i] * norms[j])
            can_rm[i] = can_rm[j] = (cos_val >= THRESH)
        for i in range(len(edges)):
            e = edges[i]
            a, b = divmod(e, 1 << 32)
            if a > b:
                continue
            edge_data[ptr, 0] = e
            edge_data[ptr, 1] = p
            edge_data[ptr, 2] = can_rm[i]
            ptr += 1
    edge_data = edge_data[:ptr]
    edge_data = edge_data[np.argsort(edge_data[:, 0], kind='mergesort')]
    E = np.unique(edge_data[:, 0])
    for i in range(len(E)):
        e = E[i]
        a = i * 10 // len(E)
        b = (i + 1) * 10 // len(E)
        if a < b:
            print(i + 1, "/", len(E))
        L = np.searchsorted(edge_data[:, 0], e)
        R = np.searchsorted(edge_data[:, 0], e + 1)
        if np.min(edge_data[L:R, 2]) == 0:
            continue
        ps = edge_data[L:R, 1]
        a, b = divmod(e, 1 << 32)
        res = [
            remove_one_edge_from_polyhedron(
                polyhedrons[p],
                a, b) for p in ps]
        ok = True
        for i in range(len(res)):
            bl = res[i][0]
            if not bl:
                ok = False
        if not ok:
            continue
        # print("remove edge", a, b)
        for i in range(len(ps)):
            p = ps[i]
            poly = res[i][1]
            polyhedrons[p] = poly
    polyhedrons, elem_conv = shrink(polyhedrons, elem_conv)
    P = len(polyhedrons)
    new_indptr = np.zeros(P + 1, np.int32)
    for p in range(P):
        new_indptr[p + 1] = len(polyhedrons[p])
    new_indptr = np.cumsum(new_indptr)
    new_dat = np.empty(new_indptr[-1], np.int32)
    for p in range(P):
        L = new_indptr[p]
        R = new_indptr[p + 1]
        new_dat[L:R] = polyhedrons[p]
    return new_indptr, new_dat


@njit
def remove_vertices(face_data_csr, node_pos, elem_conv, THRESH=0.99):
    indptr, dat = face_data_csr
    P = len(indptr) - 1
    polyhedrons = [dat[indptr[p]:indptr[p + 1]].copy() for p in range(P)]
    num_v = len(node_pos)
    nbd = np.full((num_v, 3), -1, np.int32)

    def add_nbd(a, b):
        for j in range(3):
            if nbd[a, j] == b:
                return
        for j in range(3):
            if nbd[a, j] == -1:
                nbd[a, j] = b
                return
    for p in range(P):
        poly = polyhedrons[p]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            assert k >= 3
            L = R
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                add_nbd(a, b)
                add_nbd(b, a)
    can_rm = nbd[:, 2] == -1
    for p in range(P):
        poly = polyhedrons[p].copy()
        rest = np.ones(len(poly), np.bool_)
        m = poly[0]
        L = 1
        for _ in range(m):
            L0 = L
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            assert k >= 3
            L = R
            n = len(F)
            for i in range(len(F)):
                if can_rm[F[i]]:
                    rest[L0 + 1 + i] = 0
                    n -= 1
            poly[L0] = n
            if n <= 2:
                rest[L0:R] = 0
                poly[0] -= 1
        poly = poly[rest]
        poly = polyhedrons[p]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            assert k >= 3
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for v in F:
                can_rm[v] = 0
    for p in range(P):
        poly = polyhedrons[p].copy()
        rest = np.ones(len(poly), np.bool_)
        m = poly[0]
        L = 1
        for _ in range(m):
            L0 = L
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            n = len(F)
            for i in range(len(F)):
                if can_rm[F[i]]:
                    rest[L0 + 1 + i] = 0
                    n -= 1
            poly[L0] = n
            if n == 2:
                rest[L0:R] = 0
                poly[0] -= 1
        polyhedrons[p] = poly[rest]
    polyhedrons, elem_conv = shrink(polyhedrons, elem_conv)
    P = len(polyhedrons)
    new_indptr = np.zeros(P + 1, np.int32)
    for p in range(P):
        new_indptr[p + 1] = len(polyhedrons[p])
    new_indptr = np.cumsum(new_indptr)
    new_dat = np.empty(new_indptr[-1], np.int32)
    for p in range(P):
        L = new_indptr[p]
        R = new_indptr[p + 1]
        new_dat[L:R] = polyhedrons[p]
    return new_indptr, new_dat


@njit
def check_polyhedron(poly):
    e_list = [0] * 0
    v_list = [0] * 0
    m = poly[0]
    L = 1
    for _ in range(m):
        k = poly[L]
        L += 1
        R = L + k
        F = poly[L:R]
        L = R
        if len(F) != len(np.unique(F)):
            return False
        for i in range(len(F)):
            a = F[i - 1]
            b = F[i]
            e_list.append(a << 32 | b)
            v_list.append(b)
    edges = np.unique(np.array(e_list))
    # if len(e_list) != len(edges):
    #     return False
    for e in edges:
        a, b = divmod(e, 1 << 32)
        e_rev = b << 32 | a
        i = np.searchsorted(edges, e_rev)
        if i == len(edges) or edges[i] != e_rev:
            print("ng")
            return False
    return True


@njit
def remove_vertices_2(face_data_csr, node_pos, elem_conv):
    # 近傍が 2 個しかない点は削除して辺をマージする
    indptr, dat = face_data_csr
    P = len(indptr) - 1
    polyhedrons = [dat[indptr[p]:indptr[p + 1]].copy() for p in range(P)]
    num_v = len(node_pos)
    nbd = np.full((num_v, 3), -1, np.int32)

    def add_nbd(a, b):
        for j in range(3):
            if nbd[a, j] == b:
                return
        for j in range(3):
            if nbd[a, j] == -1:
                nbd[a, j] = b
                return
    for p in range(P):
        poly = polyhedrons[p]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                add_nbd(a, b)
                add_nbd(b, a)
    can_rm = nbd[:, 2] == -1
    for p in range(P):
        poly = polyhedrons[p].copy()
        rest = np.ones(len(poly), np.bool_)
        m = poly[0]
        L = 1
        for _ in range(m):
            L0 = L
            k = poly[L]
            assert k >= 3
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            n = len(F)
            for i in range(len(F)):
                if can_rm[F[i]]:
                    rest[L0 + 1 + i] = 0
                    n -= 1
            poly[L0] = n
            if n <= 2:
                rest[L0:R] = 0
                poly[0] -= 1
        poly = poly[rest]
        ok = check_polyhedron(poly)
        assert ok
        if ok:
            polyhedrons[p] = poly
            continue
    for p in range(P):
        a = 10 * p // P
        b = 10 * (p + 1) // P
        if a < b:
            print(p + 1, "/", P)
        poly = polyhedrons[p]
        assert check_polyhedron(poly)
        cand = [0] * 0
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for v in F:
                if can_rm[v]:
                    cand.append(v)
        cand = np.unique(np.array(cand))
        for rmv in cand:
            poly = polyhedrons[p]
            newpoly = poly.copy()
            newpoly[0] = 0
            nxt = 1
            m = poly[0]
            L = 1
            for _ in range(m):
                k = poly[L]
                L += 1
                R = L + k
                F = poly[L:R]
                L = R
                F = F[F != rmv]
                if len(F) <= 2:
                    continue
                newpoly[nxt] = len(F)
                newpoly[nxt + 1:nxt + 1 + len(F)] = F
                newpoly[0] += 1
                nxt += 1 + len(F)
            poly = newpoly[:nxt]
            if not check_polyhedron(poly):
                continue
            polyhedrons[p] = poly
    polyhedrons, elem_conv = shrink(polyhedrons, elem_conv)
    P = len(polyhedrons)
    new_indptr = np.zeros(P + 1, np.int32)
    for p in range(P):
        new_indptr[p + 1] = len(polyhedrons[p])
    new_indptr = np.cumsum(new_indptr)
    new_dat = np.empty(new_indptr[-1], np.int32)
    for p in range(P):
        L = new_indptr[p]
        R = new_indptr[p + 1]
        new_dat[L:R] = polyhedrons[p]
    return new_indptr, new_dat


def print_stat(csr):
    print("data size", len(csr[1]))
    print("element count", len(csr[0]) - 1)
    count = np.sum(np.bincount(csr[1]) > 0)
    print("node count (estimated)", count)


@njit
def collect_vertex(poly):
    F = [0] * 0
    n = poly[0]
    L = 1
    for _ in range(n):
        k = poly[L]
        L += 1
        R = L + k
        F += list(poly[L:R])
        L = R
    return np.unique(np.array(F))


@njit
def merge_polyhedrons(face_data_csr, IDS, elem_conv, nxt_idx):
    indptr, dat = face_data_csr

    h0 = np.random.randint(1, 1 << 60)

    def mul(a, b):
        MASK30 = (1 << 30) - 1
        MASK31 = (1 << 31) - 1
        au, ad = a >> 31, a & MASK31
        bu, bd = b >> 31, b & MASK31
        x = ad * bu + au * bd
        xu, xd = x >> 30, x & MASK30
        x = au * bu * 2 + xu + (xd << 31) + ad * bd
        MASK61 = (1 << 61) - 1
        xu, xd = x >> 61, x & MASK61
        x = xu + xd
        if x >= MASK61:
            x -= MASK61
        return x

    def calc_face_hash(F):
        idx = np.where(F == F.min())[0][0]
        x = len(F)
        for i in range(len(F)):
            x = mul(x, h0) + (1 + F[idx - i])
            MASK61 = (1 << 61) - 1
            if x >= MASK61:
                x -= MASK61
        return x

    def collect_face_hash(p):
        res = [0] * 0
        poly = dat[indptr[p]:indptr[p + 1]]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            x = calc_face_hash(F)
            res.append(x)
            x = calc_face_hash(F[::-1])
            res.append(x)
            L = R
        return res

    def collect_edge_hash(p):
        res = [0] * 0
        poly = dat[indptr[p]:indptr[p + 1]]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                x = (a << 32) | b
                res.append(x)
            L = R
        return res
    k = 0
    n = len(IDS)
    edge_hash_list = [0] * 0
    for i in IDS:
        k += dat[indptr[i]]
        edge_hash_list += collect_edge_hash(i)
    table = np.empty((2 * k, 2), np.int64)
    t = 0
    for i in range(n):
        p = IDS[i]
        H = collect_face_hash(p)
        for x in H:
            table[t], t = (x, i), t + 1
    table = table[np.argsort(table[:, 0])]
    adj = [[0] * 0 for _ in range(len(IDS))]
    for t in range(len(table) - 1):
        if table[t, 0] != table[t + 1, 0]:
            continue
        i, j = table[t, 1], table[t + 1, 1]
        adj[i].append(j)
        adj[j].append(i)
    que = np.empty(len(IDS), np.int32)
    done = np.zeros(len(IDS), np.bool_)
    res = [[0]] * 0
    success = [0] * 0
    face_hash = np.unique(table[:, 0])
    face_count = np.zeros(len(face_hash), np.int32)

    def add(i):
        p = IDS[i]
        poly = dat[indptr[p]:indptr[p + 1]]
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            x = calc_face_hash(F)
            fid = np.searchsorted(face_hash, x)
            assert face_hash[fid] == x
            face_count[fid] += 1

    for root in range(len(IDS)):
        if done[root]:
            continue
        polyhedron_list = [root]
        done[root] = 1
        add(root)
        while True:
            cnt_before = len(polyhedron_list)
            ql = qr = 0
            for i in polyhedron_list:
                que[qr], qr = i, qr + 1
            while ql < qr:
                v, ql = que[ql], ql + 1
                for w in adj[v]:
                    if done[w]:
                        continue
                    add(w)
                    polyhedron_list.append(w)
                    done[w] = 1
                    que[qr], qr = w, qr + 1
            if len(polyhedron_list) == cnt_before:
                break
        polyhedrons = np.array(polyhedron_list)
        for i in range(len(polyhedron_list)):
            polyhedrons[i] = IDS[polyhedrons[i]]
        for p in polyhedrons:
            elem_conv[p] = nxt_idx
        nxt_idx += 1
        poly_data = [0]
        # make face data
        for p in polyhedrons:
            poly = dat[indptr[p]:indptr[p + 1]]
            m = poly[0]
            L = 1
            for _ in range(m):
                k = poly[L]
                L += 1
                R = L + k
                F = poly[L:R]
                L = R
                x = calc_face_hash(F)
                y = calc_face_hash(F[::-1])
                fidx = np.searchsorted(face_hash, x)
                fidy = np.searchsorted(face_hash, y)
                assert face_hash[fidx] == x
                cnt_y = 0
                if fidy < len(face_hash) and face_hash[fidy] == y:
                    cnt_y = face_count[fidy]
                if face_count[fidx] == cnt_y:
                    continue
                while face_count[fidx] > cnt_y:
                    face_count[fidx] -= 1
                    poly_data[0] += 1
                    poly_data.append(len(F))
                    for v in F:
                        poly_data.append(v)
        # assert check_polyhedron(np.array(poly_data))

        res.append(poly_data)
        for p in polyhedrons:
            success.append(p)
    return nxt_idx, res, success


@njit
def calc_centers(face_data_csr, node_pos):
    indptr, faces = face_data_csr
    n = len(indptr) - 1
    centers = np.empty((n, 3), np.float64)
    for p in range(n):
        P = faces[indptr[p]:indptr[p + 1]]
        V = collect_vertex(P)
        cnt = len(V)
        x, y, z = 0, 0, 0
        for v in V:
            x += node_pos[v, 0]
            y += node_pos[v, 1]
            z += node_pos[v, 2]
        centers[p] = (x / cnt, y / cnt, z / cnt)
    return centers


@njit
def merge_elements(face_data_csr, node_pos, elem_conv, K):
    elem_conv[:] = -1
    centers = calc_centers(face_data_csr, node_pos)
    n = len(centers)
    stack = [np.arange(n)]
    res_indptr = [0]
    res = [0] * 0
    prog = 0
    nxt_idx = 0
    while stack:
        IDS = stack.pop()
        if len(IDS) <= K:
            nxt_idx, polys, success = merge_polyhedrons(
                face_data_csr, IDS, elem_conv, nxt_idx)
            a = prog * 10 // len(centers)
            prog += len(success)
            b = prog * 10 // len(centers)
            if a < b:
                print(prog, "/", len(centers))
            for P in polys:
                res_indptr.append(res_indptr[-1] + len(P))
                res += P
            IDS = IDS[elem_conv[IDS] == -1]
        if len(IDS) == 0:
            continue
        lo = np.array([+np.inf, +np.inf, +np.inf])
        hi = np.array([-np.inf, -np.inf, -np.inf])
        for i in IDS:
            lo = np.minimum(lo, centers[i])
            hi = np.maximum(hi, centers[i])
        ax = np.argmax(hi - lo)
        mi = np.mean(centers[IDS, ax])
        is_sm = np.empty(len(IDS), np.bool_)
        for i in range(len(IDS)):
            is_sm[i] = centers[IDS[i], ax] < mi
        I1 = IDS[is_sm]
        I2 = IDS[~is_sm]
        stack.append(I1)
        stack.append(I2)
    return (np.array(res_indptr), np.array(res))


@njit
def merge_vertices(face_data_csr, node_pos, elem_conv, node_conv, THRESH):
    node_pos = node_pos.copy()
    indptr, dat = face_data_csr
    P = len(indptr) - 1
    polyhedrons = [dat[indptr[p]:indptr[p + 1]] for p in range(P)]

    num_v = len(node_pos)
    done = np.zeros(num_v, np.bool_)
    # v -> p
    VP = np.empty((indptr[-1], 2), np.int32)
    ptr = 0
    for p in range(P):
        poly = polyhedrons[p]
        m = poly[0]
        v_list = [0] * 0
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for v in F:
                v_list.append(v)
        vs = np.unique(np.array(v_list))
        for v in vs:
            VP[ptr, 0] = v
            VP[ptr, 1] = p
            ptr += 1
    VP = VP[:ptr]
    VP = VP[np.argsort(VP[:, 0], kind='mergesort')]
    indptr_VP = np.zeros(num_v + 1, np.int32)
    for i in range(len(VP)):
        v, p = VP[i]
        indptr_VP[v + 1] += 1
    for v in range(num_v):
        indptr_VP[v + 1] += indptr_VP[v]

    def merge(a, b):
        if done[a] or done[b]:
            return False
        dx, dy, dz = np.abs(node_pos[a] - node_pos[b])
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        if d >= THRESH:
            return False

        pa = VP[indptr_VP[a]:indptr_VP[a + 1], 1]
        pb = VP[indptr_VP[b]:indptr_VP[b + 1], 1]

        done[a] = done[b] = 1
        # print("merge", a, b)
        node_pos[a] = (node_pos[a] + node_pos[b]) / 2
        node_conv[b] = a
        for p in np.unique(np.append(pa, pb)):
            poly = polyhedrons[p].copy()
            dat = [0]
            m = poly[0]
            L = 1
            for _ in range(m):
                k = poly[L]
                L += 1
                R = L + k
                F = poly[L:R]
                L = R
                ia = ib = -1
                for i in range(len(F)):
                    if F[i] == a:
                        ia = i
                    if F[i] == b:
                        ib = i
                if ib != -1:
                    F[ib] = a
                if ia == -1 or ib == -1:
                    dat[0] += 1
                    dat.append(len(F))
                    for v in F:
                        dat.append(v)
                    continue
                if ia > ib:
                    ia, ib = ib, ia
                F1 = F[ia:ib]
                F2 = np.append(F[ib:], F[:ia])
                for F in [F1, F2]:
                    if len(F) <= 2:
                        continue
                    dat[0] += 1
                    dat.append(len(F))
                    for v in F:
                        dat.append(v)
            polyhedrons[p] = np.array(dat, np.int32)
        return True
    for p in range(P):
        poly = polyhedrons[p]
        m = poly[0]
        L = 1
        # edge_len, edge
        e_list = [(0.0, 0, 0)] * 0
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                if done[a] or done[b]:
                    continue
                d = ((node_pos[a] - node_pos[b]) ** 2).sum() ** .5
                e_list.append((d, a, b))
        e_list.sort()
        for i in range(len(e_list)):
            d, a, b = e_list[i]
            if done[a] or done[b]:
                continue
            merge(a, b)
    polyhedrons, elem_conv = shrink(polyhedrons, elem_conv)
    P = len(polyhedrons)
    new_indptr = np.zeros(P + 1, np.int32)
    for p in range(P):
        new_indptr[p + 1] = len(polyhedrons[p])
    new_indptr = np.cumsum(new_indptr)
    new_dat = np.empty(new_indptr[-1], np.int32)
    for p in range(P):
        L = new_indptr[p]
        R = new_indptr[p + 1]
        new_dat[L:R] = polyhedrons[p]
    return (new_indptr, new_dat), node_pos

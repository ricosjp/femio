import femio
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, save_npz


class MeshCompressor:
    def __init__(self, *, fem_data=None, npz=None):
        if not ((fem_data is None) ^ (npz is None)):
            raise ValueError("just one of fem_data / npz is needed")
        if fem_data is not None:
            self.fem_data = fem_data
            self.csr_raw = self.fem_data.face_data_csr()
            self.csr = self.csr_raw
            self.node_pos = fem_data.nodes.data
        else:
            # load from npz
            pass

    def compress(self, *,
                 elem, cos_thresh, dist_thresh
                 ):
        print_stat(self.csr)
        print("begin merge elements")
        K = len(self.csr[0] - 1) // elem
        self.csr = merge_elements(self.csr, self.node_pos, K)
        while True:
            before_size = len(self.csr[1])
            print_stat(self.csr)
            #print("rm vertices 1")
            # self.csr = remove_vertices_1(
            #    self.csr, self.node_pos, THRESH=cos_thresh)
            print("rm edges")
            self.csr = remove_edges(self.csr, self.node_pos, THRESH=cos_thresh)
            print("rm vertices 2")
            self.csr = remove_vertices_2(
                self.csr, self.node_pos, THRESH=cos_thresh)
            print("merge vertices")
            self.csr, self.node_pos = merge_vertices(
                self.csr, self.node_pos, THRESH=dist_thresh)
            if before_size == len(self.csr[1]):
                break
        fem_data = make_fem_data(self.csr_raw, self.node_pos, self.csr, False)
        return fem_data


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
    # return check_polyhedron(newpoly_arr), newpoly_arr


@njit
def remove_vertices_1(face_data_csr, node_pos, THRESH=0.95):
    indptr, dat = face_data_csr
    P = len(indptr) - 1
    polyhedrons = [dat[indptr[p]:indptr[p + 1]] for p in range(P)]

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
            assert k >= 3
            L = R
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                res.append(a << 32 | b)
        tmp = np.array(res)
        return np.sort(tmp)
    # vertex -> out edge
    num_v = len(node_pos)
    deg = np.zeros(num_v, np.int32)
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
            for v in F:
                deg[v] += 1
    for p in range(P):
        # 各 edge が消せるかどうかを調べる
        edges = collect_edge(p)
        poly = polyhedrons[p]
        norms = np.full((len(edges), 3), np.nan)
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
                norms[eid] = norm
        can_rm = np.empty(len(edges), np.bool_)
        for i in range(len(edges)):
            e = edges[i]
            a, b = divmod(e, 1 << 32)
            e_rev = b << 32 | a
            j = np.searchsorted(edges, e_rev)
            if j + 1 < len(edges) and edges[j] == edges[j + 1]:
                continue
            cos_val = np.sum(norms[i] * norms[j])
            can_rm[i] = can_rm[j] = (cos_val >= THRESH)
        for i in range(len(edges)):
            e = edges[i]
            if can_rm[i]:
                deg[e >> 32] -= 1
    # v -> p
    VP = np.empty((indptr[-1], 2), np.int32)
    ptr = 0
    for p in range(P):
        poly = dat[indptr[p]:indptr[p + 1]]
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
        V = np.unique(np.array(v_list))
        for v in V:
            VP[ptr], ptr = (v, p), ptr + 1
    can_rm_v = deg == 0
    VP = VP[:ptr]
    VP = VP[np.argsort(VP[:, 0], kind='mergesort')]
    indptr_VR = np.zeros(num_v + 1, np.int32)
    for i in range(ptr):
        indptr_VR[VP[i, 0] + 1] += 1
    indptr_VR = np.cumsum(indptr_VR)
    for v in range(num_v):
        if not can_rm_v[v]:
            continue
        ps = VP[indptr_VR[v]:indptr_VR[v + 1], 1]
        if len(ps) == 0:
            continue
        res = [
            remove_one_vertex_from_polyhedron(
                polyhedrons[p],
                v) for p in ps]
        ok = True
        for i in range(len(res)):
            bl = res[i][0]
            if not bl:
                ok = False
        if not ok:
            continue
        print("remove node", v)
        for i in range(len(ps)):
            p = ps[i]
            poly = res[i][1]
            polyhedrons[p] = poly
    polyhedrons = [poly for poly in polyhedrons if poly[0] > 2]
    P = len(polyhedrons)
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
            assert k >= 3
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
    # return check_polyhedron(newpoly_arr), newpoly_arr


@njit
def remove_edges(face_data_csr, node_pos, THRESH=0.99):
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
                assert edges[eid] == e
                norms[eid] = norm
        can_rm = np.empty(len(edges), np.bool_)
        for i in range(len(edges)):
            e = edges[i]
            a, b = divmod(e, 1 << 32)
            e_rev = b << 32 | a
            # print(a, b)
            j = np.searchsorted(edges, e_rev)
            if edges[j] != e_rev:
                print(a, b)
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
    # assert ptr == n
    edge_data = edge_data[:ptr]
    edge_data = edge_data[np.argsort(edge_data[:, 0], kind='mergesort')]
    for e in np.unique(edge_data[:, 0]):
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
        print("remove edge", a, b)
        for i in range(len(ps)):
            p = ps[i]
            poly = res[i][1]
            polyhedrons[p] = poly
    polyhedrons = [poly for poly in polyhedrons if poly[0] > 2]
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
def remove_vertices_2(face_data_csr, node_pos, THRESH=0.99):
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
        #ok = check_polyhedron(poly)
        # if ok:
        #    continue
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
    polyhedrons = [poly for poly in polyhedrons if poly[0] > 2]
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
def merge_polyhedrons(face_data_csr, I, hash_base, buf):
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
        x = 0
        for i in range(len(F)):
            x = mul(x, h0) + F[idx - i]
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
    n = len(I)
    edge_hash_list = [0] * 0
    for i in I:
        k += dat[indptr[i]]
        edge_hash_list += collect_edge_hash(i)
    table = np.empty((2 * k, 2), np.int64)
    t = 0
    for i in range(n):
        p = I[i]
        H = collect_face_hash(p)
        for x in H:
            table[t], t = (x, i), t + 1
    table = table[np.argsort(table[:, 0])]
    adj = [[0] * 0 for _ in range(len(I))]
    for t in range(len(table) - 1):
        if table[t, 0] != table[t + 1, 0]:
            continue
        i, j = table[t, 1], table[t + 1, 1]
        adj[i].append(j)
        adj[j].append(i)
    que = np.empty(len(I), np.int32)
    done = np.zeros(len(I), np.bool_)
    res = [[0]] * 0
    success = [0] * 0
    face_hash = np.unique(table[:, 0])
    face_count = np.zeros(len(face_hash), np.int32)

    def add(i):
        p = I[i]
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

    for root in range(len(I)):
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
            polyhedrons[i] = I[polyhedrons[i]]
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

        res.append(poly_data)
        for p in polyhedrons:
            success.append(p)
    return res, success


@njit
def merge_elements(face_data_csr, node_pos, K):
    indptr, dat = face_data_csr

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
    centers = calc_centers(face_data_csr, node_pos)
    n = len(centers)
    stack = [np.arange(n)]
    res_indptr = [0]
    res = [0] * 0
    done = np.zeros(n, np.bool_)
    hash_base = np.random.randint(0, (1 << 60) - 1, len(node_pos))
    prog = 0
    buf = np.zeros(len(node_pos), np.int32)
    while stack:
        I = stack.pop()
        if len(I) <= K:
            polys, success = merge_polyhedrons(
                face_data_csr, I, hash_base, buf)
            prog += len(success)
            print("element grouping", prog, "/", len(centers))
            for P in polys:
                res_indptr.append(res_indptr[-1] + len(P))
                res += P
            for p in success:
                done[p] = 1
            I = I[~done[I]]
        if len(I) == 0:
            continue
        # print("K", K, "I", I)
        lo = np.array([+np.inf, +np.inf, +np.inf])
        hi = np.array([-np.inf, -np.inf, -np.inf])
        for i in I:
            lo = np.minimum(lo, centers[i])
            hi = np.maximum(hi, centers[i])
        ax = np.argmax(hi - lo)
        mi = np.mean(centers[I, ax])
        is_sm = np.empty(len(I), np.bool_)
        for i in range(len(I)):
            is_sm[i] = centers[I[i], ax] < mi
        I1 = I[is_sm]
        I2 = I[~is_sm]
        stack.append(I1)
        stack.append(I2)
    assert np.all(buf == 0)
    return (np.array(res_indptr), np.array(res))


@njit
def reindex(node_pos, face_data_csr):
    indptr, faces = face_data_csr
    isin = np.zeros(len(node_pos), np.bool_)
    for f in range(len(indptr) - 1):
        P = faces[indptr[f]:indptr[f + 1]]
        n = P[0]
        L = 1
        for _ in range(n):
            k = P[L]
            L += 1
            R = L + k
            for i in range(L, R):
                isin[P[i]] = 1
            L = R
        assert L == len(P)
    idx = np.cumsum(isin) - 1
    for f in range(len(indptr) - 1):
        P = faces[indptr[f]:indptr[f + 1]]
        n = P[0]
        L = 1
        for _ in range(n):
            k = P[L]
            L += 1
            R = L + k
            for i in range(L, R):
                P[i] = idx[P[i]]
            L = R
    node_pos = node_pos[isin]
    return node_pos, face_data_csr, isin


def make_fem_data(original_csr, node_pos, csr, make_mat=True):
    csr = (csr[0].copy(), csr[1].copy())
    N = len(node_pos)
    node_pos, csr, isin = reindex(node_pos, csr)
    indptr, faces = csr
    P = len(indptr) - 1
    nodes = femio.FEMAttribute(
        'nodes',
        ids=np.arange(len(node_pos)) + 1,
        data=node_pos)
    element_data = np.empty(P, object)
    for p in range(P):
        element_data[p] = collect_vertex(faces[indptr[p]:indptr[p + 1]])
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
                        'face', ids=np.arange(P) + 1, data=face_data_list)})})
    if not make_mat:
        print("nodes", len(fem_data.nodes.data))
        print("elements", len(fem_data.elements.data))
        return fem_data
    mat_1, mat_2 = calculate_convert_matrix(N, original_csr, isin)
    return fem_data, mat_1, mat_2


"""
@njit
def merge_vertices(face_data_csr, node_pos, THRESH):
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

    def can_merge(p, a, b):
        poly = polyhedrons[p]
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
            if ia == -1 or ib == -1:
                continue
            d1 = abs(ia - ib)
            d2 = len(F) - d1
            if min(d1, d2) > 1:
                print(a, b, F)
                return False
        return True

    def collect_edge(a, b):
        # a に入る辺、a から出る辺を集める
        FRM = [0] * 0
        TO = [0] * 0
        ps = VP[indptr_VP[a]: indptr_VP[a + 1], 1]
        for p in ps:
            poly = polyhedrons[p]
            m = poly[0]
            L = 1
            for _ in range(m):
                k = poly[L]
                L += 1
                R = L + k
                F = poly[L:R]
                L = R
                if len(F) <= 3 and a in F and b in F:
                    continue
                for i in range(len(F)):
                    if F[i] == a:
                        FRM.append(F[i - 1])
                    if F[i - 1] == a:
                        TO.append(F[i])
        return np.unique(np.array(FRM)), np.unique(np.array(TO))

    def merge(a, b):
        if done[a] or done[b]:
            return False
        dx, dy, dz = np.abs(node_pos[a] - node_pos[b])
        if max(dx, dy, dz) >= THRESH:
            return False

        pa = VP[indptr_VP[a]:indptr_VP[a + 1], 1]
        pb = VP[indptr_VP[b]:indptr_VP[b + 1], 1]

        for p in pa:
            if p in pb and not can_merge(p, a, b):
                # print("reject bad poly")
                return False
        FRM_a, TO_a = collect_edge(a, b)
        FRM_b, TO_b = collect_edge(b, a)
        for v in FRM_a:
            if v in FRM_b:
                #print("reject common v", a, b, v)
                return False
        for v in TO_a:
            if v in TO_b:
                #print("reject common v", a, b, v)
                return False
        done[a] = done[b] = 1
        print("merge", a, b)
        node_pos[a] = (node_pos[a] + node_pos[b]) / 2
        for p in np.unique(np.append(pa, pb)):
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
                for i in range(len(F)):
                    if F[i] == b:
                        F[i] = a
                for i in range(len(F)):
                    if F[i - 1] == F[i]:
                        rest[L0 + 1 + i] = 0
                        poly[L0] -= 1
                if poly[L0] <= 2:
                    rest[L0:R] = 0
                    poly[0] -= 1
            polyhedrons[p] = poly[rest]
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
            if d > THRESH:
                break
            if done[a] or done[b]:
                continue
            merge(a, b)
    polyhedrons = [poly for poly in polyhedrons if poly[0] > 2]
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
"""


@njit
def merge_vertices(face_data_csr, node_pos, THRESH):
    """
    非常に短い辺を縮約して、中点に置き換える
    """
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

    def collect_edge(a, b):
        # a に入る辺、a から出る辺を集める
        FRM = [0] * 0
        TO = [0] * 0
        ps = VP[indptr_VP[a]: indptr_VP[a + 1], 1]
        for p in ps:
            poly = polyhedrons[p]
            m = poly[0]
            L = 1
            for _ in range(m):
                k = poly[L]
                L += 1
                R = L + k
                F = poly[L:R]
                L = R
                if len(F) <= 3 and a in F and b in F:
                    continue
                for i in range(len(F)):
                    if F[i] == a:
                        FRM.append(F[i - 1])
                    if F[i - 1] == a:
                        TO.append(F[i])
        return np.unique(np.array(FRM)), np.unique(np.array(TO))

    def merge(a, b):
        if done[a] or done[b]:
            return False
        dx, dy, dz = np.abs(node_pos[a] - node_pos[b])
        if max(dx, dy, dz) >= THRESH:
            return False

        pa = VP[indptr_VP[a]:indptr_VP[a + 1], 1]
        pb = VP[indptr_VP[b]:indptr_VP[b + 1], 1]

        done[a] = done[b] = 1
        print("merge", a, b)
        node_pos[a] = (node_pos[a] + node_pos[b]) / 2
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
            # if d > sz * THRESH:
            #    break
            if done[a] or done[b]:
                continue
            merge(a, b)
    polyhedrons = [poly for poly in polyhedrons if poly[0] > 2]
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

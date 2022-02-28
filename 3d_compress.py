import femio
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix, save_npz


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
def check_polyhedron(poly):
    e_list = [0] * 0
    v_list = [0] * 0
    m = poly[0]
    if m <= 2:
        return False
    L = 1
    for _ in range(m):
        k = poly[L]
        if k <= 2:
            return False
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
    if len(e_list) != len(edges):
        return False
    for e in edges:
        a, b = divmod(e, 1 << 32)
        e_rev = b << 32 | a
        i = np.searchsorted(edges, e_rev)
        if i == len(edges) or edges[i] != e_rev:
            return False
    return True
    """
    cnt_v = len(np.unique(np.array(v_list)))
    cnt_e = len(edges) // 2
    cnt_f = m
    return cnt_v - cnt_e + cnt_f == 2
    """


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


def to_csr(face_data_list):
    indptr = [len(row) for row in face_data_list]
    indptr = np.append(0, np.cumsum(indptr))
    return (indptr, np.concatenate(face_data_list))


@njit
def calculate_nbd_in(N, csr, isin, knn):
    # nodal graph を作る
    G_li = [(0, 0)] * 0
    indptr, face_data = csr
    P = len(indptr) - 1
    for p in range(P):
        V = collect_vertex(face_data[indptr[p]:indptr[p + 1]])
        for v in V:
            G_li.append((v, p))
    G_ve = np.array(G_li)
    G_ev = G_ve[:, ::-1].copy()
    G_ve = G_ve[np.argsort(G_ve[:, 0])]
    G_ev = G_ev[np.argsort(G_ev[:, 0])]
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
        if isin[v]:
            nbd[v, 0] = v
            que[qr] = (v, v)
            qr += 1
    while ql < qr:
        v, frm = que[ql]
        ql += 1
        for i in range(indptr_ve[v], indptr_ve[v + 1]):
            e = G_ve[i, 1]
            for j in range(indptr_ev[e], indptr_ev[e + 1]):
                to = G_ev[j, 1]
                success = False
                for k in range(knn):
                    if nbd[to, k] == -1:
                        nbd[to, k] = frm
                        break
                    if nbd[to, k] == frm:
                        break
                if not success:
                    continue
                que[qr] = (to, frm)
                qr += 1
    return nbd


def calculate_convert_matrix(N, csr, isin, knn=3):
    nbd = calculate_nbd_in(N, csr, isin, knn)
    M = np.sum(isin)
    cols = np.where(isin)[0]
    rows = np.arange(M)
    vals = np.ones(M, np.bool_)
    mat1 = csr_matrix((vals, (rows, cols)), (M, N), dtype=np.bool_)
    rows = np.empty(knn * N, np.int32)
    for k in range(knn):
        rows[k::3] = np.arange(N)
    nbd[isin, 1:] = -1
    cols = nbd.ravel()
    idx = cols != -1
    V = np.where(isin)[0]
    rows = rows[idx]
    cols = cols[idx]
    cols = np.searchsorted(V, cols)
    vals = np.ones(len(rows), np.bool_)
    mat2 = csr_matrix((vals, (rows, cols)), (N, M), dtype=np.bool_)
    return mat1, mat2


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
        return fem_data
    mat_1, mat_2 = calculate_convert_matrix(N, original_csr, isin)
    return fem_data, mat_1, mat_2


@njit
def merge_polyhedrons(face_data_csr, I, hash_base, buf, check_euler):
    indptr, dat = face_data_csr

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
            x = 0
            for v in F:
                x ^= hash_base[v]
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
    cnt_e = 0
    n = len(I)
    edge_hash_list = [0] * 0
    for i in I:
        k += dat[indptr[i]]
        edge_hash_list += collect_edge_hash(i)
    table = np.empty((k, 2), np.int64)
    t = 0
    for i in range(n):
        p = I[i]
        H = collect_face_hash(p)
        for x in H:
            table[t], t = (x, i), t + 1
    assert t == k
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
    edge_hash = np.unique(np.array(edge_hash_list, np.int64))
    face_hash = np.unique(table[:, 0])
    edge_count = np.zeros(len(edge_hash), np.int32)
    vertex_count = buf
    face_count = np.zeros(len(face_hash), np.int32)
    cnt_v = cnt_e = cnt_f = 0
    cnt_bad_e = 0

    def add_face(fid, F):
        nonlocal cnt_v, cnt_e, cnt_f, cnt_bad_e
        assert face_count[fid] == 0
        face_count[fid] = 1
        cnt_f += 1
        for v in F:
            if vertex_count[v] == 0:
                cnt_v += 1
            vertex_count[v] += 1
        for i in range(len(F)):
            a, b = F[i - 1], F[i]
            e = np.searchsorted(edge_hash, (a << 32) | b)
            if edge_count[e] == 0:
                cnt_e += 1
            if edge_count[e] == 1:
                cnt_bad_e += 1
            edge_count[e] += 1

    def rm_face(fid, F):
        nonlocal cnt_v, cnt_e, cnt_f, cnt_bad_e
        assert face_count[fid] == 1
        face_count[fid] = 0
        cnt_f -= 1
        for v in F:
            vertex_count[v] -= 1
            if vertex_count[v] == 0:
                cnt_v -= 1
        for i in range(len(F)):
            b, a = F[i - 1], F[i]
            e = np.searchsorted(edge_hash, (a << 32) | b)
            edge_count[e] -= 1
            if edge_count[e] == 1:
                cnt_bad_e -= 1
            if edge_count[e] == 0:
                cnt_e -= 1

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
            x = 0
            for v in F:
                x ^= hash_base[v]
            fid = np.searchsorted(face_hash, x)
            assert face_hash[fid] == x
            if face_count[fid] == 0:
                add_face(fid, F)
            else:
                rm_face(fid, F)
    for root in range(len(I)):
        if done[root]:
            continue
        polyhedron_list = [root]
        done[root] = 1
        add(root)
        assert cnt_bad_e == 0
        euler = cnt_v - cnt_e // 2 + cnt_f
        if check_euler and euler != 2:
            continue
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
                    euler = cnt_v - (cnt_e // 2) + cnt_f
                    ng = cnt_bad_e > 0
                    if check_euler and euler != 2:
                        ng = True
                    if ng:
                        add(w)
                        continue
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
                x = 0
                for v in F:
                    x ^= hash_base[v]
                fid = np.searchsorted(face_hash, x)
                assert face_hash[fid] == x
                if face_count[fid] == 0:
                    continue
                rm_face(fid, F)
                poly_data[0] += 1
                poly_data.append(len(F))
                for v in F:
                    poly_data.append(v)
        if cnt_v or cnt_e or cnt_f or cnt_bad_e:
            cnt_v = cnt_e = cnt_f = cnt_bad_e = 0
            continue
        assert cnt_v == cnt_e == cnt_f == 0
        assert cnt_bad_e == 0
        res.append(poly_data)
        for p in polyhedrons:
            success.append(p)
    return res, success


@njit
def grouping_and_merge(face_data_csr, node_pos, K, check_euler=True):
    centers = calc_centers(face_data_csr, node_pos)
    n = len(centers)
    stack = [np.arange(n)]
    res_indptr = [0]
    res = [0] * 0
    done = np.zeros(n, np.bool_)
    hash_base = np.random.randint(0, (1 << 63) - 1, len(node_pos))
    prog = 0
    prev = 0
    buf = np.zeros(len(node_pos), np.int32)
    while stack:
        I = stack.pop()
        if len(I) <= K:
            polys, success = merge_polyhedrons(
                face_data_csr, I, hash_base, buf, check_euler)
            prog += len(success)
            if prog > prev + (len(centers) // 100):
                print("element grouping", prog, "/", len(centers))
                prev = prog
            for P in polys:
                res_indptr.append(res_indptr[-1] + len(P))
                res += P
            for p in success:
                done[p] = 1
            I = I[~done[I]]
        if len(I) == 0:
            continue
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
def grouping_and_merge_suzuki(face_data_csr, node_pos, K, check_euler=True):
    centers = calc_centers(face_data_csr, node_pos)
    X, Y, Z = centers.T
    inner = np.abs(X) <= 4
    inner &= np.abs(Y) <= 2
    inner &= np.abs(Z) <= 2
    stack_outer = [np.where(~inner)[0]]
    stack_inner = stack_outer[:]
    stack_inner.clear()
    I = np.where(inner)[0]
    # I を分割
    sz = 0.08
    for i in range(100):
        xl = sz * (i - 50)
        xr = xl + sz
        if i == 0:
            xl -= 1
        if i == 99:
            xr += 1
        Ix = I[(xl <= X[I]) & (X[I] < xr)]
        for j in range(50):
            yl = sz * (j - 25)
            yr = yl + sz
            if j == 0:
                yl -= 1
            if j == 49:
                yr += 1
            Iy = Ix[(yl <= Y[Ix]) & (Y[Ix] < yr)]
            for k in range(25):
                zl = sz * k
                zr = zl + sz
                if k == 0:
                    zl = -1
                if k == 24:
                    zr += 1
                J = Iy[(zl <= Z[Iy]) & (Z[Iy] < zr)]
                stack_inner.append(J)
    res_indptr = [0]
    res = [0] * 0
    n = len(csr[0]) - 1
    done = np.zeros(n, np.bool_)
    hash_base = np.random.randint(0, (1 << 63) - 1, len(node_pos))
    prog = 0
    prev = 0
    buf = np.zeros(len(node_pos), np.int32)
    stack = stack_outer
    while stack:
        I = stack.pop()
        lo = np.array([+np.inf, +np.inf, +np.inf])
        hi = np.array([-np.inf, -np.inf, -np.inf])
        for i in I:
            lo = np.minimum(lo, centers[i])
            hi = np.maximum(hi, centers[i])
        if len(I) <= K or (hi - lo).max() <= 0.08:
            polys, success = merge_polyhedrons(
                face_data_csr, I, hash_base, buf, check_euler)
            prog += len(success)
            if prog > prev + (len(centers) // 100):
                print("element grouping", prog, "/", len(centers))
                prev = prog
            for P in polys:
                res_indptr.append(res_indptr[-1] + len(P))
                res += P
            for p in success:
                done[p] = 1
            I = I[~done[I]]
        if len(I) == 0:
            continue
        ax = np.argmax(hi - lo)
        mi = np.mean(centers[I, ax])
        is_sm = np.empty(len(I), np.bool_)
        for i in range(len(I)):
            is_sm[i] = centers[I[i], ax] < mi
        I1 = I[is_sm]
        I2 = I[~is_sm]
        stack.append(I1)
        stack.append(I2)
    stack = stack_inner
    while stack:
        I = stack.pop()
        polys, success = merge_polyhedrons(
            face_data_csr, I, hash_base, buf, check_euler)
        prog += len(success)
        if prog > prev + (len(centers) // 100):
            print("element grouping", prog, "/", len(centers))
            prev = prog
        for P in polys:
            res_indptr.append(res_indptr[-1] + len(P))
            res += P
        for p in success:
            done[p] = 1
        I = I[~done[I]]
        if len(I) == 0:
            continue
        stack.append(I)
    assert np.all(buf == 0)
    return (np.array(res_indptr), np.array(res))


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
            L = R
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                res.append(a << 32 | b)
        return np.unique(np.array(res))
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
            assert edges[j] == e_rev
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
        return np.unique(np.array(res))
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
    assert ptr == n
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
        if ok:
            continue
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
def check(csr):
    indptr, dat = csr
    for p in range(len(indptr) - 1):
        poly = dat[indptr[p]:indptr[p + 1]]
        e_list = [0] * 0
        v_list = [0] * 0
        m = poly[0]
        if m <= 2:
            print("m <= 2", p)
            return False
        L = 1
        for _ in range(m):
            k = poly[L]
            if k < 3:
                print("k<3", p)
                return False
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            if len(F) != len(np.unique(F)):
                print("non-simple polygon", p)
                return False
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                e_list.append(a << 32 | b)
                v_list.append(b)
        edges = np.unique(np.array(e_list))
        if len(e_list) != len(edges):
            print("multiple edge", p)
            return False
        for e in edges:
            a, b = divmod(e, 1 << 32)
            e_rev = b << 32 | a
            i = np.searchsorted(edges, e_rev)
            if i == len(edges) or edges[i] != e_rev:
                print("not contain rev_edge", p, divmod(a, b))
                return False
        """cnt_v = len(np.unique(np.array(v_list)))
        cnt_e = len(edges) // 2
        cnt_f = m
        euler = cnt_v - cnt_e + cnt_f
        if euler != 2:
            print("euler chara is wrong", p, euler)"""
    return True


def print_stat(csr):
    print("data size", len(csr[1]))
    print("element count", len(csr[0]) - 1)
    count = np.sum(np.bincount(csr[1]) > 0)
    print("node count (estimated)", count)


def save(csr):
    indptr, dat = csr
    count = np.sum(np.bincount(dat) > 0)
    np.save(f'csr_indptr_{count}.npy', indptr)
    np.save(f'csr_dat_{count}.npy', dat)


@njit
def cut_elem_indices(csr, I):
    indptr, dat = csr
    sz = np.zeros(len(I), np.int32)
    for i in range(len(I)):
        sz[i] = indptr[I[i] + 1] - indptr[I[i]]
    new_indptr = np.append(0, np.cumsum(sz))
    new_dat = np.empty(new_indptr[-1], np.int32)
    for i in range(len(I)):
        L = new_indptr[i]
        R = new_indptr[i + 1]
        new_dat[L:R] = dat[indptr[I[i]]:indptr[I[i] + 1]]
    new_csr = (new_indptr, new_dat)
    return new_csr


def cut_region(csr_raw, centers, xlim, ylim, zlim):
    X, Y, Z = centers.T
    in_X = (xlim[0] <= X) & (X <= xlim[1])
    in_Y = (ylim[0] <= Y) & (Y <= ylim[1])
    in_Z = (zlim[0] <= Z) & (Z <= zlim[1])
    I = np.where(in_X & in_Y & in_Z)[0]
    print("elem count", len(I))
    return cut_elem_indices(csr_raw, I)


"""
fem_data = femio.read_files('polyvtk', 'cut_data_3.vtu')
csr = to_csr(fem_data.elemental_data['face']['polyhedron'].data)
node_pos = fem_data.nodes.data

csr_raw = np.load('csr_raw.npz')
csr_raw = (csr_raw['indptr'], csr_raw['dat'])
node_pos = np.load('node_pos.npy')
centers = calc_centers(csr_raw,node_pos)
import random
for i in range(100):
    while 1:
        x = -4 + random.random() * 8.00
        y = -2 + random.random() * 4.00
        z = 2 * random.random()
        xlim = (x, x + 0.08)
        ylim = (y, y+0.08)
        zlim = (z, z+0.08)
        cut_csr = cut_region(csr_raw,centers,xlim,ylim,zlim)
        n = len(cut_csr[0]) - 1
        if n > 100:
            break
    cut_data = make_fem_data(csr_raw,node_pos,cut_csr,False)
    name = f"cut_data_{i}.vtu"
    cut_data.write('polyvtk', name, overwrite=True)
"""


"""
@njit
def remove_edges_2(face_data_csr, node_pos, rm_len, rm_width):
    indptr, dat = face_data_csr
    P = len(indptr) - 1
    polyhedrons = [dat[indptr[p]:indptr[p + 1]].copy() for p in range(P)]

    def calc_area(F):
        vc = np.zeros(3)
        for i in range(2, len(F)):
            vc1 = node_pos[F[1]] - node_pos[F[0]]
            vc2 = node_pos[F[i]] - node_pos[F[0]]
            vc += np.cross(vc1, vc2)
        return (vc * vc).sum() ** .5

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
        return np.unique(np.array(res))
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
        can_rm = np.empty(len(edges), np.bool_)
        poly = polyhedrons[p]
        width = np.empty(len(edges))
        m = poly[0]
        L = 1
        for _ in range(m):
            k = poly[L]
            L += 1
            R = L + k
            F = poly[L:R]
            L = R
            area = calc_area(F)
            for i in range(len(F)):
                a = F[i - 1]
                b = F[i]
                e = a << 32 | b
                eid = np.searchsorted(edges, e)
                elen = ((node_pos[a] - node_pos[b])**2).sum() ** .5
                can_rm[eid] = elen <= rm_len and area / elen <= rm_width
        for i in range(len(edges)):
            e = edges[i]
            a, b = divmod(e, 1 << 32)
            e_rev = b << 32 | a
            j = np.searchsorted(edges, e_rev)
            if can_rm[i] == 1 or can_rm[j] == 1:
                can_rm[i] = can_rm[j] = 1
        for i in range(len(edges)):
            e = edges[i]
            a, b = divmod(e, 1 << 32)
            if a > b:
                continue
            edge_data[ptr, 0] = e
            edge_data[ptr, 1] = p
            edge_data[ptr, 2] = can_rm[i]
            ptr += 1
    assert ptr == n
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
                return False
        return True

    def collect_edge(a):
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
                for i in range(len(F)):
                    if F[i] == a:
                        FRM.append(F[i - 1])
                    if F[i - 1] == a:
                        TO.append(F[i])
        return np.unique(np.array(FRM)), np.unique(np.array(TO))

    def is_inner(a):
        x, y, z = node_pos[a]
        return abs(x) <= 4.0 and abs(y) <= 2.0 and abs(z) <= 2.0

    def merge(a, b):
        if not (is_inner(a) and is_inner(b)):
            return False
        if done[a] or done[b]:
            return False
        pa = VP[indptr_VP[a]: indptr_VP[a + 1], 1]
        pb = VP[indptr_VP[b]: indptr_VP[b + 1], 1]
        for p in pa:
            if p in pb and not can_merge(p, a, b):
                return False
        FRM_a, TO_a = collect_edge(a)
        FRM_b, TO_b = collect_edge(b)
        for v in FRM_a:
            if v in FRM_b:
                return False
        for v in TO_a:
            if v in TO_b:
                return False
        done[a] = done[b] = 1
        print("merge vertices", a, b)
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
fem_data = femio.read_files('polyvtk', 'cut_data_4.vtu')
csr_raw = to_csr(fem_data.elemental_data['face']['polyhedron'].data)
csr=csr_raw
node_pos = fem_data.nodes.data
csr = grouping_and_merge(csr,node_pos,1<<30,True)
while 1:
    THRESH=0.99
    now = len(csr[1])
    csr, node_pos = merge_vertices(csr, node_pos, 0.003)
    assert check(csr)
    for _ in range(3):
        print_stat(csr)
        csr_tmp = (csr[0].copy(), csr[1].copy())
        csr = remove_vertices_1(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        csr_tmp = (csr[0].copy(), csr[1].copy())
        csr = remove_edges(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        csr_tmp = (csr[0].copy(), csr[1].copy())
        csr = remove_vertices_2(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        print_stat(csr)
    new_fem_data = make_fem_data(csr_raw,node_pos,csr,0)
    count = new_fem_data.nodes.data.shape[0]
    name = f"merge_v_{count}.vtu"
    new_fem_data.write('polyvtk', name, overwrite=True)
    if len(csr[1]) == now:
        break
"""


"""
K = 30
THRESH = 0.90
csr = grouping_and_merge(csr, node_pos, K, True)
print_stat(csr)
assert check(csr)
save(csr)
while True:
    while True:
        sz = len(csr[1])
        print_stat(csr)
        print("remove vertices")
        csr_tmp = (csr[0].copy(), csr[1].copy())
        csr = remove_vertices_1(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        save(csr)
        print_stat(csr)
        print("remove edges")
        csr = remove_edges(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        print_stat(csr)
        save(csr)
        print("remove vertices")
        csr = remove_vertices_2(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        save(csr)
        if len(csr[1]) == sz:
            break
    new_fem_data, mat_1, mat_2 = make_fem_data(csr_raw, node_pos, csr)
    count = len(new_fem_data.nodes.data)
    print("節点数, 要素数")
    print(len(new_fem_data.nodes.data), len(new_fem_data.elements.data))
    name = f"compressed_data_{count}.vtu"
    new_fem_data.write('polyvtk', name, overwrite=True)
    name = f"csr_matrix_1_{count}.npz"
    save_npz(name, mat_1)
    name = f"csr_matrix_2_{count}.npz"
    save_npz(name, mat_2)
    if count < 10**6:
        break
    csr_tmp = (csr[0].copy(), csr[1].copy())
    K = 5
    while True:
        csr = grouping_and_merge(csr, node_pos, K, False)
        if check(csr):
            save(csr)
            break
        csr = csr_tmp
        K += 1
"""

"""
K = 100000
csr = csr_raw
csr = grouping_and_merge(csr, node_pos, K, 1)
print_stat(csr)
assert check(csr)

THRESH = 0.95
sz = len(csr[1])
csr_tmp = (csr[0].copy(), csr[1].copy())

print_stat(csr)
csr_tmp = (csr[0].copy(), csr[1].copy())
csr = remove_vertices_1(csr, node_pos, THRESH)
if not check(csr):
    csr = csr_tmp

csr_tmp = (csr[0].copy(), csr[1].copy())
csr = remove_edges(csr, node_pos, THRESH)
if not check(csr):
    csr = csr_tmp

csr_tmp = (csr[0].copy(), csr[1].copy())
csr = remove_vertices_2(csr, node_pos, THRESH)
if not check(csr):
    csr = csr_tmp

print_stat(csr)


print_stat(csr)
new_fem_data, mat_1, mat_2 = make_fem_data(csr_raw, node_pos, csr)
new_fem_data.write('polyvtk', 'outer.vtu')"""

"""
node_pos = np.load('node_pos.npy')
csr_raw = np.load('csr_raw.npz')
node_pos = np.load('node_pos.npy')
csr_raw = (csr_raw['indptr'], csr_raw['dat'])
csr = (csr_raw[0].copy(), csr_raw[1].copy())
"""


def main():
    print("begin read file")
    fem_data = femio.read_files(
        'polyvtk',
        '/home/group/ricos/data/suzuki/20211018/ESCUDO/EnsightGold/with_u.vtu')
    print("begin to_polyhedron")
    fem_data = fem_data.to_polyhedron()
    node_pos = fem_data.nodes.data
    csr_raw = to_csr(fem_data.elemental_data['face']['polyhedron'].data)
    csr = (csr_raw[0].copy(), csr_raw[1].copy())

    csr = grouping_and_merge_suzuki(csr_raw, node_pos, 100, True)
    while True:
        THRESH = 0.99
        now = len(csr[1])
        assert check(csr)
        print_stat(csr)
        csr_tmp = (csr[0].copy(), csr[1].copy())
        csr = remove_vertices_1(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        csr_tmp = (csr[0].copy(), csr[1].copy())
        csr = remove_edges(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        csr_tmp = (csr[0].copy(), csr[1].copy())
        csr = remove_vertices_2(csr, node_pos, THRESH)
        if not check(csr):
            csr = csr_tmp
        print_stat(csr)
        if len(csr[1]) == now:
            break

    edge_len_thresh = 0.008
    while True:
        edge_len_thresh += 0.001
        THRESH = 0.95
        while True:
            print_stat(csr)
            now = len(csr[1])
            print("merge edge len", edge_len_thresh)
            csr_tmp = (csr[0].copy(), csr[1].copy())
            csr, node_pos = merge_vertices(csr, node_pos, edge_len_thresh)
            if not check(csr):
                csr = csr_tmp
            csr_tmp = (csr[0].copy(), csr[1].copy())
            csr = remove_vertices_1(csr, node_pos, THRESH)
            if not check(csr):
                csr = csr_tmp
            csr_tmp = (csr[0].copy(), csr[1].copy())
            csr = remove_edges(csr, node_pos, THRESH)
            if not check(csr):
                csr = csr_tmp
            csr_tmp = (csr[0].copy(), csr[1].copy())
            csr = remove_vertices_2(csr, node_pos, THRESH)
            if not check(csr):
                csr = csr_tmp
            if len(csr[1]) == now:
                break
        dat = make_fem_data(csr_raw, node_pos, csr, 0)
        count = len(dat.nodes.data)
        name = f'xxx_{count}.vtu'
        dat.write('polyvtk', name, overwrite=True)
        if count <= 10 ** 6:
            return make_fem_data(csr_raw, node_pos, csr, 1)


new_fem_data, mat_1, mat_2 = main()

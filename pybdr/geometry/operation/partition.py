import numpy as np
from pybdr.geometry import *
from .convert import cvt2


def __interval2interval(src: Interval, r: float):
    def __ll2arr(ll, fill_value: float):
        lens = [lst.shape[0] for lst in ll]
        max_len = max(lens)
        mask = np.arange(max_len) < np.array(lens)[:, None]
        arr = np.ones((len(lens), max_len, 2), dtype=float) * fill_value
        arr[mask] = np.concatenate(ll)
        return arr, mask

    def __get_seg(dim_idx: int, seg_num: int):
        if seg_num <= 1:
            return np.array([src.inf[dim_idx], src.sup[dim_idx]], dtype=float).reshape(
                (1, -1)
            )
        else:
            samples = np.linspace(src.inf[dim_idx], src.sup[dim_idx], num=seg_num + 1)
            this_segs = np.zeros((seg_num, 2), dtype=float)
            this_segs[:, 0] = samples[:-1]
            this_segs[:, 1] = samples[1:]
            return this_segs

    assert len(src.shape) == 1
    nums = np.floor((src.sup - src.inf) / r).astype(dtype=int) + 1
    segs, _ = __ll2arr([__get_seg(i, nums[i]) for i in range(src.shape[0])], np.nan)
    idx_list = [np.arange(nums[i]) for i in range(src.shape[0])]
    ext_idx = np.array(np.meshgrid(*idx_list)).T.reshape((-1, len(idx_list)))
    aux_idx = np.tile(np.arange(src.shape[0]), ext_idx.shape[0])
    bounds = segs[aux_idx, ext_idx.reshape(-1)].reshape((-1, src.shape[0], 2))
    return [Interval(bound[:, 0], bound[:, 1]) for bound in bounds]


def __interval2zonotope(src: Interval, r: float):
    parts = __interval2interval(src, r)
    return [cvt2(part, Geometry.TYPE.ZONOTOPE) for part in parts]


def __zonotope2interval(src: Interval, r: float):
    return [cvt2(zono, Geometry.TYPE.INTERVAL) for zono in __zonotope2zonotope(src, r)]


def __zonotope2zonotope(src: Zonotope, r: float):
    def __linearly_independent_base(x: np.ndarray, rank: int):
        from itertools import combinations

        ind = np.asarray(list(combinations(np.arange(x.shape[1]), rank)))
        subx = x.T[ind]
        rank = np.linalg.matrix_rank(subx)
        return ind[rank >= rank]

    def __boundary_matrix(gen):
        def __matrix_cross_product(x: np.ndarray):
            from itertools import combinations

            # only care about n by n-1 matrix
            assert x.ndim == 2 and x.shape[0] == x.shape[1] + 1
            # generate appropriate indices
            ind = np.asarray(list(combinations(np.arange(x.shape[0]), x.shape[1])))
            ind = ind[np.argsort(np.sum(ind, axis=-1))[::-1]]
            # extract sub-matrices
            subx = x[ind, :]
            # compute det for all sub-matrices
            dets = np.linalg.det(subx)
            coeffs = np.power(-1, np.arange(ind.shape[0]))
            # return the final results
            return coeffs * dets

        def __is_valid_base(bounds, bound_gen):
            if len(bounds) <= 0:
                return True
            ind = [np.setdiff1d(bound_gen, bound).size <= 0 for bound in bounds]
            return not np.any(ind)

        def __boundary_row(gx: np.ndarray, com: np.ndarray, bounds: list):
            if not __is_valid_base(bounds, com):
                return None, None
            # else
            remain_ind = np.setdiff1d(np.arange(gx.shape[1]), com)
            mcp = __matrix_cross_product(gx[:, com])
            inn_prod = np.dot(mcp, gx[:, remain_ind])
            gtz_mask = inn_prod > 0
            ltz_mask = inn_prod < 0
            ez_mask = np.logical_not(ltz_mask | gtz_mask)
            col_ind = np.union1d(com, remain_ind[ez_mask])
            sym_rows = np.zeros((2, gx.shape[1]))  # symmetric rows
            sym_rows[:, col_ind] = -2
            sym_rows[0, remain_ind[gtz_mask]] = -1
            sym_rows[0, remain_ind[ltz_mask]] = 1
            sym_rows[1, remain_ind[gtz_mask]] = 1
            sym_rows[1, remain_ind[ltz_mask]] = -1
            return sym_rows, col_ind

        bound_cols = []
        boundary_rows = []
        combs = __linearly_independent_base(gen, gen.shape[0] - 1)
        for comb in combs:
            rows, cols = __boundary_row(gen, comb, bound_cols)
            if rows is None:
                continue
            bound_cols.append(cols)
            boundary_rows.append(rows)

        return np.concatenate(boundary_rows, axis=0)

    def __part_matrix(x: np.ndarray, d: int):
        pm = []
        b = x
        for i in range(x.shape[1] - d):
            b = b[b[:, i] != -2, :]
            ind = b[:, i] == -1
            t = b[ind, :]
            t[:, i] = -2
            b[:, i] = 1
            pm.append(t)
        b_last_row = b[-1, :]
        b_last_row[x.shape[1] - d :] = -2
        pm.append(b_last_row.reshape((1, -1)))
        return np.concatenate(pm, axis=0)

    def __restore_parts(x: np.ndarray, cx: np.ndarray, gx: np.ndarray):
        parts = []
        for i in range(x.shape[0]):
            mask_append = x[i, :] == -2
            mask_sub = x[i, :] == -1
            mask_add = x[i, :] == 1
            gen = gx[:, mask_append]
            c = cx + gx[:, mask_add].sum(axis=-1) - gx[:, mask_sub].sum(axis=-1)
            parts.append(Zonotope(c, gen))
        return parts

    dim = src.gen.shape[0]
    # check the rank of generator matrix
    gen_rank = np.linalg.matrix_rank(src.gen)
    if gen_rank < dim:
        raise NotImplementedError("not full rank zonotope")

    lib = __linearly_independent_base(src.gen, dim)
    remain_ind = np.setdiff1d(np.arange(src.gen_num), lib[0])
    gen_new = src.gen[:, np.append(remain_ind, lib[0])]
    bound_m = __boundary_matrix(gen_new)
    part_m = __part_matrix(bound_m, dim)
    return __restore_parts(part_m, src.c, gen_new)


def partition(src: Geometry.Base, r: float, elem: Geometry.TYPE):
    if src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.INTERVAL:
        return __interval2interval(src, r)
    elif src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.ZONOTOPE:
        return __interval2zonotope(src, r)
    elif src.type == Geometry.TYPE.ZONOTOPE and elem == Geometry.TYPE.INTERVAL:
        return __zonotope2interval(src, r)
    elif src.type == Geometry.TYPE.ZONOTOPE and elem == Geometry.TYPE.ZONOTOPE:
        return __zonotope2zonotope(src, r)
    else:
        raise NotImplementedError

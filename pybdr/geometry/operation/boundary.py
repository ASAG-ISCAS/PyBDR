from itertools import chain

import numpy as np

from pybdr.geometry import Geometry, Interval, Polytope, Zonotope
from pybdr.util.functional import CSPSolver, RealPaver

from .convert import cvt2


def _interval2interval(src: Interval, r: float):
    bd = []
    assert len(src.shape) == 1
    dims = np.arange(src.shape[0])
    for i in range(src.shape[0]):
        valid_dims = np.setdiff1d(dims, i)
        g = src.proj(valid_dims).grid(r)
        data = np.zeros((g.shape[0], src.shape[0] * 2, 2), dtype=float)
        # set this dimension inf related boundary
        data[:, valid_dims, :] = g
        data[:, i, :] = src.inf[i]
        # set this dimension sup related boundary
        data[:, valid_dims + src.shape[0], :] = g
        data[:, i + src.shape[0], :] = src.sup[i]
        data = data.reshape((-1, src.shape[0], 2))
        bd.append([Interval(cur_data[:, 0], cur_data[:, 1]) for cur_data in data])

    return list(chain.from_iterable(bd))


def _interval2polytope(src: Interval, r: float):
    bound_boxes = _interval2interval(src, r)
    return [cvt2(box, Geometry.TYPE.POLYTOPE) for box in bound_boxes]


def _interval2zonotope(src: Interval, r: float):
    bd_intervals = _interval2interval(src, r)
    return [cvt2(interval, Geometry.TYPE.ZONOTOPE) for interval in bd_intervals]


def _polytope2interval(src: Polytope, r: float):
    def f(x):
        z = Interval.zeros(src.a.shape[0])
        for i in range(src.a.shape[0]):
            for j in range(src.a.shape[1]):
                z[i] += src.a[i, j] * x[j]
            z[i] -= src.b[i]

        ind0 = np.logical_and(z.inf <= 0, z.sup >= 0)
        ind1 = z.inf > 0
        sum = np.sum(ind0)
        return 0 < sum and np.sum(ind1) <= 0

    lb = np.min(src.vertices, axis=0) - r
    ub = np.max(src.vertices, axis=0) + r
    boxes = CSPSolver.solve(f, lb, ub, r)
    return boxes


def _polytope2interval_realpaver(src: Polytope, r: float):
    # using realpaver to get the boundary
    realpaver = RealPaver()
    num_const, num_var = src.a.shape

    # fix bugs when part of the boundary parallel to some axis，make it slightly tilted
    def refine_const(arr: np.array, tol: float):
        ind = np.where((np.abs(arr)) <= tol)
        arr[ind] += tol
        return arr

    mat_a = refine_const(src.a, 1e-14)
    vec_b = refine_const(src.b, 1e-14)  # 1e-14 may vary for different problem, need to refine # TODO

    #  get the domain of the variables
    bound_vs = src.vertices
    bound_inf = np.min(bound_vs, axis=0) - 100 * r
    bound_sup = np.max(bound_vs, axis=0) + 100 * r

    # set variables
    for idx in range(num_var):
        realpaver.add_variable("x" + str(idx), bound_inf[idx], bound_sup[idx], "[", "]")

    # set constraints
    for idx_const in range(num_const):
        this_const = ""
        for idx_var in range(num_var):
            this_const += (
                    "{:.20e}".format(mat_a[idx_const, idx_var]) + "*x" + str(idx_var) + "+"
            )
        this_const = this_const[:-1] + "<=" + "{:.20e}".format(vec_b[idx_const])
        realpaver.add_constraint(this_const)

    realpaver.set_branch(precision=r)
    boxes = realpaver.solve()
    bound_boxes = []
    for b in boxes:
        if b[0] == "OUTER":
            bound_boxes.append((b[2]))
    return bound_boxes


def _polytope2polytope(src: Polytope, r: float):
    bound_boxes = _polytope2interval(src, r)
    return [cvt2(box, Geometry.TYPE.POLYTOPE) for box in bound_boxes]


def _polytope2zonotope(src: Polytope, r: float):
    boxes = _polytope2interval(src, r)
    return [cvt2(box, Geometry.TYPE.ZONOTOPE) for box in boxes]


def _zonotope2zonotope(src: Zonotope, r: float):
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

    def __linearly_independent_base(x: np.ndarray):
        from itertools import combinations

        ind = np.asarray(list(combinations(np.arange(x.shape[1]), x.shape[0] - 1)))
        subx = x.T[ind]
        rank = np.linalg.matrix_rank(subx)
        return ind[rank >= x.shape[0] - 1]

    def __is_valid_base(bounds, bound_gen):
        if len(bounds) <= 0:
            return True
        ind = [np.setdiff1d(bound_gen, bound).size <= 0 for bound in bounds]
        return not np.any(ind)

    def __boundary_gen(cx: np.ndarray, gx: np.ndarray, com: np.ndarray, bounds: list):
        if not __is_valid_base(bounds, com):
            return None, None
        # else
        sym_bounds = []
        remain_ind = np.setdiff1d(np.arange(gx.shape[1]), com)
        mcp = __matrix_cross_product(gx[:, com])
        inn_prod = np.dot(mcp, gx[:, remain_ind])
        gtz_mask = inn_prod > 0
        ltz_mask = inn_prod < 0
        ez_mask = np.logical_not(ltz_mask | gtz_mask)
        col_ind = np.union1d(com, remain_ind[ez_mask])
        gen = gx[:, col_ind]
        c_gtz = gx[:, remain_ind[gtz_mask]].sum(axis=-1)
        c_ltz = gx[:, remain_ind[ltz_mask]].sum(axis=-1)
        sym_bounds.append(Zonotope(cx + c_gtz - c_ltz, gen))
        sym_bounds.append(Zonotope(cx - c_gtz + c_ltz, gen))
        return sym_bounds, col_ind

    dim = src.gen.shape[0]
    # check the rank of generator matrix
    gen_rank = np.linalg.matrix_rank(src.gen)
    if gen_rank < dim:
        return src  # not full rank zonotope in N dimensional space, boundary is itself
    # else we try to extract the boundary of this zonotope in the form of zonotope
    bound_cols = []
    boundaries = []
    combs = __linearly_independent_base(src.gen)
    for comb in combs:
        bounds, cols = __boundary_gen(src.c, src.gen, comb, bound_cols)
        if bounds is None:
            continue
        bound_cols.append(cols)
        boundaries.extend(bounds)
    return boundaries


def _zonotope2interval(src: Zonotope, r: float):
    return _polytope2interval_realpaver(cvt2(src, Geometry.TYPE.POLYTOPE), r)


def _zonotope2polytope(src: Zonotope, r: float):
    bound_boxes = _zonotope2interval(src, r)
    return [cvt2(box, Geometry.TYPE.POLYTOPE) for box in bound_boxes]


def boundary(src: Geometry.Base, r: float, elem: Geometry.TYPE):
    if src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.INTERVAL:
        return _interval2interval(src, r)
    elif src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.POLYTOPE:
        return _interval2polytope(src, r)
    elif src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.ZONOTOPE:
        return _interval2zonotope(src, r)
    elif src.type == Geometry.TYPE.POLYTOPE and elem == Geometry.TYPE.INTERVAL:
        return _polytope2interval_realpaver(src, r)
    elif src.type == Geometry.TYPE.POLYTOPE and elem == Geometry.TYPE.POLYTOPE:
        return _polytope2polytope(src, r)
    elif src.type == Geometry.TYPE.POLYTOPE and elem == Geometry.TYPE.ZONOTOPE:
        return _polytope2zonotope(src, r)
    elif src.type == Geometry.TYPE.ZONOTOPE and elem == Geometry.TYPE.INTERVAL:
        return _zonotope2interval(src, r)
    elif src.type == Geometry.TYPE.ZONOTOPE and elem == Geometry.TYPE.POLYTOPE:
        return _zonotope2polytope(src, r)
    elif src.type == Geometry.TYPE.ZONOTOPE and elem == Geometry.TYPE.ZONOTOPE:
        return _zonotope2zonotope(src, r)
    else:
        raise NotImplementedError

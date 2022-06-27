from itertools import chain

import numpy as np

from pyrat.geometry import *
from pyrat.util.functional import CSPSolver
from .convert import cvt2


def __interval2interval(src: Interval, r: float):
    bd = []
    assert len(src.dim) == 1
    dims = np.arange(src.dim[0])
    for i in range(src.dim[0]):
        valid_dims = np.setdiff1d(dims, i)
        g = src.proj(valid_dims).grid(r)
        data = np.zeros((g.shape[0], src.dim[0] * 2, 2), dtype=float)
        # set this dimension inf related boundary
        data[:, valid_dims, :] = g
        data[:, i, :] = src.inf[i]
        # set this dimension sup related boundary
        data[:, valid_dims + src.dim[0], :] = g
        data[:, i + src.dim[0], :] = src.sup[i]
        data = data.reshape((-1, src.dim[0], 2))
        bd.append([Interval(cur_data[:, 0], cur_data[:, 1]) for cur_data in data])

    return list(chain.from_iterable(bd))


def __interval2zonotope(src: Interval, r: float):
    bd_intervals = __interval2interval(src, r)
    return [cvt2(interval, Geometry.TYPE.ZONOTOPE) for interval in bd_intervals]


def __polytope2interval(src: Polytope, r: float):
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


def __polytope2zonotope(src: Polytope, r: float):
    boxes = __polytope2interval(src, r)
    return [cvt2(box, Geometry.TYPE.ZONOTOPE) for box in boxes]


def boundary(src: Geometry.Base, r: float, elem: Geometry.TYPE):
    if src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.INTERVAL:
        return __interval2interval(src, r)
    elif src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.ZONOTOPE:
        return __interval2zonotope(src, r)
    elif src.type == Geometry.TYPE.POLYTOPE and elem == Geometry.TYPE.INTERVAL:
        return __polytope2interval(src, r)
    elif src.type == Geometry.TYPE.POLYTOPE and elem == Geometry.TYPE.ZONOTOPE:
        return __polytope2zonotope(src, r)
    else:
        raise NotImplementedError

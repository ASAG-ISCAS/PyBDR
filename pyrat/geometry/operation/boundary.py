import numpy as np
from itertools import chain
from pyrat.geometry import *
from pyrat.geometry.operation import cvt2


def __interval2interval(src: Interval, r: float):
    bd = []
    dims = np.arange(src.dim)
    for i in range(src.dim):
        valid_dims = np.setdiff1d(dims, i)
        g = src.proj(valid_dims).grid(r)
        data = np.zeros((g.shape[0], src.dim * 2, 2), dtype=float)
        # set this dimension inf related boundary
        data[:, valid_dims, :] = g
        data[:, i, :] = src.inf[i]
        # set this dimension sup related boundary
        data[:, valid_dims + src.dim, :] = g
        data[:, i + src.dim, :] = src.sup[i]
        data = data.reshape((-1, src.dim, 2))
        bd.append([Interval(cur_data[:, 0], cur_data[:, 1]) for cur_data in data])

    return list(chain.from_iterable(bd))


def __interval2zonotope(src: Interval, r: float):
    bd_intervals = __interval2interval(src, r)
    return [cvt2(interval, Geometry.TYPE.ZONOTOPE) for interval in bd_intervals]


def boundary(src: Geometry.Base, r: float, elem: Geometry.TYPE):
    if src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.INTERVAL:
        return __interval2interval(src, r)
    elif src.type == Geometry.TYPE.INTERVAL and elem == Geometry.TYPE.ZONOTOPE:
        return __interval2zonotope(src, r)
    else:
        raise NotImplementedError

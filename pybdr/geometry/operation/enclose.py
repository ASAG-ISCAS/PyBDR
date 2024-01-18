from pybdr.geometry import *
import numpy as np
from .convert import cvt2


def _zz2z(lhs: Zonotope, rhs: Zonotope):
    # get generator numbers
    lhs_num, rhs_num = lhs.gen_num + 1, rhs.gen_num + 1
    # if first zonotope has more or equal generators
    z_cut, z_add, z_eq = None, None, None
    if rhs_num < lhs_num:
        z_cut = lhs.z[:, :rhs_num]
        z_add = lhs.z[:, rhs_num:lhs_num]
        z_eq = rhs.z
    else:
        z_cut = rhs.z[:, :lhs_num]
        z_add = rhs.z[:, lhs_num:rhs_num]
        z_eq = lhs.z
    z = np.concatenate([(z_cut + z_eq) * 0.5, (z_cut - z_eq) * 0.5, z_add], axis=1)
    return Zonotope(z[:, 0], z[:, 1:])


def _ii2i(lhs: Interval, rhs: Interval):
    inf = np.minimum(lhs.inf, rhs.inf)
    sup = np.maximum(lhs.sup, rhs.sup)
    return Interval(inf, sup)


def enclose(lhs: Geometry.Base, rhs: Geometry.Base, target: Geometry.TYPE):
    if lhs.type == Geometry.TYPE.INTERVAL and rhs.type == Geometry.TYPE.INTERVAL:
        if target == Geometry.TYPE.INTERVAL:
            return _ii2i(lhs, rhs)
        elif target == Geometry.TYPE.POLYTOPE:
            return cvt2(_ii2i(lhs, rhs), Geometry.TYPE.POLYTOPE)
        elif target == Geometry.TYPE.ZONOTOPE:
            return cvt2(_ii2i(lhs, rhs), Geometry.TYPE.ZONOTOPE)
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.INTERVAL and rhs.type == Geometry.TYPE.POLYTOPE:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.INTERVAL and rhs.type == Geometry.TYPE.ZONOTOPE:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.POLYTOPE and rhs.type == Geometry.TYPE.INTERVAL:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.POLYTOPE and rhs.type == Geometry.TYPE.POLYTOPE:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.POLYTOPE and rhs.type == Geometry.TYPE.ZONOTOPE:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.ZONOTOPE and rhs.type == Geometry.TYPE.INTERVAL:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.ZONOTOPE and rhs.type == Geometry.TYPE.POLYTOPE:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif lhs.type == Geometry.TYPE.ZONOTOPE and rhs.type == Geometry.TYPE.ZONOTOPE:
        if target == Geometry.TYPE.INTERVAL:
            raise NotImplementedError
        elif target == Geometry.TYPE.POLYTOPE:
            raise NotImplementedError
        elif target == Geometry.TYPE.ZONOTOPE:
            return _zz2z(lhs, rhs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

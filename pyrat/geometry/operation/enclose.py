from pyrat.geometry import *
import numpy as np


def __zono_by_zono(lhs: Zonotope, rhs: Zonotope, target: Geometry.TYPE):
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


def enclose(lhs: Geometry.Base, rhs: Geometry.Base, target: Geometry.TYPE = None):
    target = lhs.type if target is None else target
    if lhs.type == Geometry.TYPE.ZONOTOPE and rhs.type == Geometry.TYPE.ZONOTOPE:
        return __zono_by_zono(lhs, rhs, target)
    else:
        raise NotImplementedError

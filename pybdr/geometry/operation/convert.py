from __future__ import annotations

import numpy as np
import pypoman
from scipy.spatial import ConvexHull
import pybdr.util.functional.auxiliary as aux
from pybdr.geometry import *


def _interval2interval(source: Interval):
    return source


def _interval2polytope(source: Interval):
    assert len(source.shape) == 1
    a = np.concatenate([np.eye(source.shape[0]), -np.eye(source.shape[0])], axis=0)
    b = np.concatenate([source.sup, -source.inf])
    return Polytope(a, b)


def _interval2zonotope(source: Interval):
    c = source.c
    gen = np.diag(source.rad)
    return Zonotope(c, gen)


def _polytope2interval(source: Polytope):
    inf = np.min(source.vertices, axis=0)
    sup = np.max(source.vertices, axis=0)
    return Interval(inf, sup)


def _polytope2polytope(source: Polytope):
    pass


def _polytope2zonotpe(source: Polytope):
    pass


def _zonotope2interval(source: Zonotope):
    """
    over approximate a zonotope by an interval
    :param source: given zonotope
    :return:
    """
    c = source.c
    # determine left and right limit, specially designed for high performance
    delta = np.sum(abs(source.z), axis=1) - abs(c)
    left_limit = c - delta
    right_limit = c + delta
    return Interval(left_limit, right_limit)


def _zonotope2polytope(source: Zonotope):
    return cvt2(source.vertices, Geometry.TYPE.POLYTOPE)


def _zonotope2zonotope(source: Zonotope):
    return source


# def _zonotope2polyzonotope(source: Zonotope):
#     # TODO
#     raise NotImplementedError


def _vertices2interval(source: np.ndarray):
    assert source.ndim == 2
    inf = np.min(source, axis=0)
    sup = np.max(source, axis=0)
    return Interval(inf, sup)


def _vertices2polytope_old(source: np.ndarray):
    a, b = pypoman.compute_polytope_halfspaces(source)
    return Polytope(a, b)


def _vertices2polytope(source: np.ndarray):
    print(source.shape)
    convex_hull = ConvexHull(source)
    eq = convex_hull.equations
    return Polytope(eq[:, :-1], -eq[:, -1])


def _vertices2zonotope(source: np.ndarray):
    raise NotImplementedError


# def _polyzonotope2zonotope(source: PolyZonotope):
#     if not aux.is_empty(source.gen):
#         # determine dependent generators with exponents that are all even
#         temp = np.prod(np.ones_like(source) - np.mod(source.exp_mat, 2), 1)
#         g_quad = source.gen[:, temp == 1]
#
#         # compute zonotope parameters
#         c = source.c + 0.5 * np.sum(g_quad, axis=1)
#         gen = np.concatenate(
#             [
#                 source.gen[:, temp == 0],
#                 0.5 * g_quad,
#                 np.zeros((source.shape, 0))
#                 if source.gen_rst is None
#                 else source.gen_rst,
#             ],
#             axis=1,
#         )
#
#         # generate zonotope
#         return Zonotope(c, gen)
#
#     else:
#         return Zonotope(source.c, source.gen_rst)


def _cvt_from_vertices(vertices: np.ndarray, target: Geometry.TYPE):
    if target == Geometry.TYPE.INTERVAL:
        return _vertices2interval(vertices)
    elif target == Geometry.TYPE.POLYTOPE:
        return _vertices2polytope(vertices)
    elif target == Geometry.TYPE.ZONOTOPE:
        return _vertices2zonotope(vertices)
    else:
        raise NotImplementedError


def _cvt_from_geometry(src: Geometry.Base, target: Geometry.TYPE):
    if src.type == Geometry.TYPE.INTERVAL and target == Geometry.TYPE.INTERVAL:
        return _interval2interval(src)
    elif src.type == Geometry.TYPE.INTERVAL and target == Geometry.TYPE.POLYTOPE:
        return _interval2polytope(src)
    elif src.type == Geometry.TYPE.INTERVAL and target == Geometry.TYPE.ZONOTOPE:
        return _interval2zonotope(src)
    elif src.type == Geometry.TYPE.POLYTOPE and target == Geometry.TYPE.INTERVAL:
        return _polytope2interval(src)
    elif src.type == Geometry.TYPE.POLYTOPE and target == Geometry.TYPE.POLYTOPE:
        return src
    elif src.type == Geometry.TYPE.POLYTOPE and target == Geometry.TYPE.ZONOTOPE:
        raise NotImplementedError
    elif src.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.INTERVAL:
        return _zonotope2interval(src)
    elif src.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.POLYTOPE:
        return _zonotope2polytope(src)
    elif src.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.ZONOTOPE:
        return _zonotope2zonotope(src)
    else:
        raise NotImplementedError


def cvt2(src, target: Geometry.TYPE):
    if src is None:
        return src
    elif isinstance(src, np.ndarray):
        return _cvt_from_vertices(src, target)
    elif isinstance(src, Geometry.Base):
        return _cvt_from_geometry(src, target)
    else:
        raise NotImplementedError


def cvt2_old(src, target: Geometry.TYPE):
    if src is None:
        return src
    elif isinstance(src, np.ndarray) and target == Geometry.TYPE.INTERVAL:
        return _vertices2interval(src)
    elif isinstance(src, np.ndarray) and target == Geometry.TYPE.POLYTOPE:
        return _vertices2polytope(src)
    elif isinstance(src, Geometry.Base):
        if src.type == target:
            return src
        elif src.type == Geometry.TYPE.INTERVAL and target == Geometry.TYPE.ZONOTOPE:
            return _interval2zonotope(src)
        elif src.type == Geometry.TYPE.INTERVAL and target == Geometry.TYPE.POLYTOPE:
            return _interval2polytope(src)
        elif src.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.INTERVAL:
            return _zonotope2interval(src)
        elif src.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.POLYTOPE:
            return _zonotope2polytope(src)
        elif (
                src.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.POLY_ZONOTOPE
        ):
            return _zonotope2polyzonotope(src)
        elif (
                src.type == Geometry.TYPE.POLY_ZONOTOPE and target == Geometry.TYPE.ZONOTOPE
        ):
            return _polyzonotope2zonotope(src)
    else:
        raise NotImplementedError

from __future__ import annotations

import numpy as np
import pypoman
from scipy.spatial import ConvexHull
import pyrat.util.functional.auxiliary as aux
from pyrat.geometry import *


def _interval2zonotope(source: Interval):
    c = source.c
    gen = np.diag(source.rad)
    return Zonotope(c, gen)


def _interval2polytope(source: Interval):
    assert len(source.shape) == 1
    a = np.concatenate([np.eye(source.shape[0]), -np.eye(source.shape[0])], axis=0)
    b = np.concatenate([source.sup, -source.inf])
    return Polytope(a, b)


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
    source.remove_zero_gen()
    c, g = source.c, source.gen
    n, n_gen = source.shape, source.gen_num

    is_deg = 0
    if np.linalg.matrix_rank(g) >= n:
        if n > 1:
            raise NotImplementedError
        else:
            c = 1

    # TODO
    raise NotImplementedError


def _zonotope2polyzonotope(source: Zonotope):
    # TODO
    raise NotImplementedError


def _vertices2interval(source: np.ndarray):
    assert source.ndim == 2
    inf = np.min(source, axis=0)
    sup = np.max(source, axis=0)
    return Interval(inf, sup)


def _vertices2polytope_old(source: np.ndarray):
    a, b = pypoman.compute_polytope_halfspaces(source)
    # ensure the inequality
    diff = a[None, :, :] @ source[:, :, None] - b[None, :, None]
    # return Polytope(a, b + np.max(diff))
    return Polytope(a, b)


def _vertices2polytope(source: np.ndarray):
    convex_hull = ConvexHull(source)
    eq = convex_hull.equations
    return Polytope(eq[:, :2], -eq[:, -1])


def _polyzonotope2zonotope(source: PolyZonotope):
    if not aux.is_empty(source.gen):
        # determine dependent generators with exponents that are all even
        temp = np.prod(np.ones_like(source) - np.mod(source.exp_mat, 2), 1)
        g_quad = source.gen[:, temp == 1]

        # compute zonotope parameters
        c = source.c + 0.5 * np.sum(g_quad, axis=1)
        gen = np.concatenate(
            [
                source.gen[:, temp == 0],
                0.5 * g_quad,
                np.zeros((source.shape, 0)) if source.gen_rst is None else source.gen_rst,
            ],
            axis=1,
        )

        # generate zonotope
        return Zonotope(c, gen)

    else:
        return Zonotope(source.c, source.gen_rst)


def cvt2(source, target: Geometry.TYPE):
    if source is None:
        return source
    elif isinstance(source, np.ndarray) and target == Geometry.TYPE.INTERVAL:
        return _vertices2interval(source)
    elif isinstance(source, np.ndarray) and target == Geometry.TYPE.POLYTOPE:
        return _vertices2polytope(source)
    elif isinstance(source, Geometry.Base):
        if source.type == target:
            return source
        elif source.type == Geometry.TYPE.INTERVAL and target == Geometry.TYPE.ZONOTOPE:
            return _interval2zonotope(source)
        elif source.type == Geometry.TYPE.INTERVAL and target == Geometry.TYPE.POLYTOPE:
            return _interval2polytope(source)
        elif source.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.INTERVAL:
            return _zonotope2interval(source)
        elif source.type == Geometry.TYPE.ZONOTOPE and target == Geometry.TYPE.POLYTOPE:
            return _zonotope2polytope(source)
        elif (
                source.type == Geometry.TYPE.ZONOTOPE
                and target == Geometry.TYPE.POLY_ZONOTOPE
        ):
            return _zonotope2polyzonotope(source)
        elif (
                source.type == Geometry.TYPE.POLY_ZONOTOPE
                and target == Geometry.TYPE.ZONOTOPE
        ):
            return _polyzonotope2zonotope(source)
    else:
        raise NotImplementedError

"""
Boundary extraction of initial sets with interval analysis, based on codac.
"""

from __future__ import annotations

import codac
import numpy as np


def _to_box(domain) -> codac.IntervalVector:
    """convert a pybdr Interval (or a codac IntervalVector) to a new codac IntervalVector"""
    from pybdr.geometry import Interval

    if isinstance(domain, codac.IntervalVector):
        return codac.IntervalVector(domain)
    if isinstance(domain, Interval):
        return codac.IntervalVector([[float(lo), float(up)] for lo, up in zip(domain.inf, domain.sup)])
    raise TypeError("domain must be a pybdr Interval or a codac IntervalVector")


def _bounds(box: codac.IntervalVector) -> np.ndarray:
    """(n, 2) array of the lower and upper bounds, reading components is faster than np.array(box.lb())"""
    return np.array([[box[i].lb(), box[i].ub()] for i in range(box.size())])


def _to_interval(box: codac.IntervalVector):
    from pybdr.geometry import Interval

    bounds = _bounds(box)
    return Interval(bounds[:, 0], bounds[:, 1])


def _pave(domain: codac.IntervalVector, ctc, eps: float) -> list[codac.IntervalVector]:
    """bisect the domain into boxes of width <= eps that can not be removed by the contractor"""
    result = []
    stack = [domain]
    while stack:
        box = stack.pop()
        ctc.contract(box)
        if box.is_empty():
            continue
        if box.max_diam() <= eps:
            result.append(box)
            continue
        left, right = box.bisect(box.max_diam_index())
        stack.append(right)
        stack.append(left)
    return result


def polytope_boundary(a: np.ndarray, b: np.ndarray, eps: float, domain=None) -> list:
    """
    boxes (as pybdr Intervals) of width <= eps covering the boundary of the polytope {x | a x <= b}

    :param a: constraint matrix
    :param b: constraint vector
    :param eps: maximal width of the boxes
    :param domain: box to search in (pybdr Interval or codac IntervalVector), defaults to the bounding box
    :return: list of Intervals
    """
    from pybdr.geometry import Interval
    from pybdr.geometry.polytope import halfspaces_to_vertices

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if domain is None:
        vs = halfspaces_to_vertices(a, b)
        domain = Interval(vs.min(axis=0) - eps, vs.max(axis=0) + eps)

    # one contractor per constraint, so that only the constraints still active in a box are applied
    x = codac.VectorVar(a.shape[1])
    ctcs = [codac.CtcInverse(codac.AnalyticFunction([x], codac.Matrix(a[[i]]) * x - codac.Vector(b[[i]])),
                             codac.IntervalVector([[-np.inf, 0.0]]))
            for i in range(a.shape[0])]
    abs_a = np.abs(a)
    # margin against rounding errors when classifying constraints with floating point arithmetic
    tol = 1e-12 * max(1.0, float(np.max(np.abs(b))))

    result = []
    stack = [(_to_box(domain), np.arange(a.shape[0]))]
    while stack:
        box, active = stack.pop()
        bounds = _bounds(box)
        c, r = bounds.mean(axis=1), (bounds[:, 1] - bounds[:, 0]) / 2
        # exact range of the linear constraints a_i x - b_i over the box
        mid = a[active] @ c - b[active]
        rad = abs_a[active] @ r
        if np.any(mid - rad > tol):
            continue  # some constraint is violated in the whole box
        active = active[mid + rad >= -tol]  # drop constraints satisfied in the whole box
        if active.size == 0:
            continue  # the box is in the interior
        for i in active:
            ctcs[i].contract(box)
            if box.is_empty():
                break
        if box.is_empty():
            continue
        if box.max_diam() <= eps:
            result.append(box)
            continue
        left, right = box.bisect(box.max_diam_index())
        stack.append((right, active))
        stack.append((left, active))

    return [_to_interval(box) for box in result]


def function_boundary(f: codac.AnalyticFunction_Scalar, domain, eps: float,
                      simplify_result: bool = True, simplify_eps: float = 0.005) -> list:
    """
    boxes (as pybdr Intervals) of width <= eps covering the zero level set {x | f(x) = 0} within the domain

    :param f: scalar codac AnalyticFunction
    :param domain: box to search in (pybdr Interval or codac IntervalVector)
    :param eps: maximal width of the boxes
    :param simplify_result: re-pave isolated boxes with simplify_eps to remove spurious boxes
    :param simplify_eps: maximal width of the boxes when re-paving isolated boxes
    :return: list of Intervals
    """
    ctc = codac.CtcInverse(f, [0.0])
    result = _pave(_to_box(domain), ctc, eps)

    if simplify_result:
        for box in _isolated_boxes(result):
            result.remove(box)
            result += _pave(codac.IntervalVector(box), ctc, simplify_eps)

    return [_to_interval(box) for box in result]


def extract_boundary(init_interval, init_set, eps: float = 0.04,
                     simplify_result: bool = True, simplify_eps: float = 0.005):
    """
    boxes covering the boundary of the initial set, converted to zonotopes

    :param init_interval: box to search in (pybdr Interval or codac IntervalVector), optional for
        polytopes and zonotopes
    :param init_set: Polytope, Zonotope, or scalar codac AnalyticFunction f describing the boundary f(x) = 0
    :param eps: maximal width of the boxes
    :param simplify_result: only for AnalyticFunction, see function_boundary
    :param simplify_eps: only for AnalyticFunction, see function_boundary
    :return: list of Zonotopes
    """
    from pybdr.geometry import Geometry, Polytope, Zonotope
    from pybdr.geometry.operation.convert import cvt2

    # codac.AnalyticFunction is a factory since codac 2.1, scalar functions are AnalyticFunction_Scalar
    if isinstance(init_set, codac.AnalyticFunction_Scalar):
        boxes = function_boundary(init_set, init_interval, eps, simplify_result, simplify_eps)
    elif isinstance(init_set, (Polytope, Zonotope)):
        if isinstance(init_set, Zonotope):
            init_set = cvt2(init_set, Geometry.TYPE.POLYTOPE)
        boxes = polytope_boundary(init_set.a, init_set.b, eps, init_interval)
    else:
        raise NotImplementedError()
    return [cvt2(box, Geometry.TYPE.ZONOTOPE) for box in boxes]


def _boxes_touch(b1: codac.IntervalVector, b2: codac.IntervalVector, tol=1e-15):
    n = b1.size()
    for i in range(n):
        if b1[i].ub() < b2[i].lb() - tol or b2[i].ub() < b1[i].lb() - tol:
            return False
    return True


def _isolated_boxes(boxes, tol=1e-12):
    isolated = []
    for i, b in enumerate(boxes):
        connected = False
        for j, b2 in enumerate(boxes):
            if i == j:
                continue
            if _boxes_touch(b, b2, tol):
                connected = True
                break
        if not connected:
            isolated.append(b)
    return isolated

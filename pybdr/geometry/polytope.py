from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

import pybdr.util.functional.auxiliary as aux
from .geometry import Geometry

if TYPE_CHECKING:
    from .zonotope import Zonotope


def _unique_rows(x: np.ndarray, decimals: int = 10) -> np.ndarray:
    """remove rows that coincide up to rounding, keeping the original values and order"""
    scale = max(1.0, float(np.max(np.abs(x))))
    _, idx = np.unique(np.round(x / scale, decimals), axis=0, return_index=True)
    return x[np.sort(idx)]


def chebyshev_center(a: np.ndarray, b: np.ndarray):
    """
    center and radius of the largest ball inscribed in {x | a x <= b}
    """
    n = a.shape[1]
    norms = np.linalg.norm(a, axis=1)
    # maximize r s.t. a_i x + ||a_i|| r <= b_i, r >= 0
    res = linprog(
        c=np.append(np.zeros(n), -1.0),
        A_ub=np.hstack([a, norms[:, None]]),
        b_ub=b,
        bounds=[(None, None)] * n + [(0, None)],
        method="highs",
    )
    if res.status == 2:
        raise ValueError("polytope is empty")
    if res.status == 3:
        raise ValueError("polytope is unbounded")
    if not res.success:
        raise RuntimeError("chebyshev center computation failed: " + res.message)
    return res.x[:n], res.x[n]


def halfspaces_to_vertices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    vertices of the bounded, full-dimensional polytope {x | a x <= b}
    """
    c, r = chebyshev_center(a, b)
    if r <= 1e-12 * max(1.0, float(np.max(np.abs(b)))):
        raise ValueError("polytope has empty interior, vertices can not be enumerated")
    if a.shape[1] == 1:
        # qhull does not support 1-dimensional input, the polytope is an interval here
        return np.array([[np.min(b[a[:, 0] > 0] / a[a[:, 0] > 0, 0])], [np.max(b[a[:, 0] < 0] / a[a[:, 0] < 0, 0])]])
    with np.errstate(divide="ignore", invalid="ignore"):
        vs = HalfspaceIntersection(np.hstack([a, -b[:, None]]), c).intersections
    if not np.all(np.isfinite(vs)):
        raise ValueError("polytope is unbounded")
    return _unique_rows(vs)


def vertices_to_halfspaces(vs: np.ndarray):
    """
    halfspace representation (a, b) with a x <= b of the convex hull of the given points
    """
    eq = _unique_rows(ConvexHull(vs).equations)
    return eq[:, :-1], -eq[:, -1]


class Polytope(Geometry.Base):
    def __init__(self, a: np.ndarray, b: np.ndarray):
        """
        aX<=b
        """
        assert not aux.is_empty(a)
        assert not aux.is_empty(b)
        assert a.ndim == 2 and a.shape[0] > 0
        assert b.ndim == 1 and a.shape[0] == b.shape[0]
        self._a = a.astype(dtype=float)
        self._b = b.astype(dtype=float)
        self._c = None
        self._vs = None
        self._type = Geometry.TYPE.POLYTOPE

    # =============================================== property
    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def c(self) -> np.ndarray:
        """
        chebyshev center of this polytope
        """
        if self._c is None:
            self._c, _ = chebyshev_center(self._a, self._b)
        return self._c

    @property
    def shape(self) -> int:
        assert not self.is_empty
        return self._a.shape[1]

    @property
    def is_empty(self) -> bool:
        return aux.is_empty(self._a) and aux.is_empty(self._b)

    @property
    def vertices(self) -> np.ndarray:
        """
        get extreme vertices of this polytope defined as AX<=B
        """
        if self._vs is None:
            self._vs = halfspaces_to_vertices(self._a, self._b)
        return self._vs

    @property
    def info(self):
        info = "\n ----------------- Polytope BEGIN -----------------\n"
        info += ">>> dimension -- constraints num\n"
        info += str(self.shape) + "\n"
        info += str(self._a.shape[0]) + "\n"
        info += "\n ----------------- Polytope END -----------------\n"
        return info

    @property
    def type(self) -> Geometry.TYPE:
        return self._type

    # =============================================== operator

    def __contains__(self, item):
        def __contains_pts(pts: np.ndarray):
            assert pts.ndim == 1 or pts.ndim == 2
            if pts.ndim == 1:
                if self.shape != pts.shape[0]:
                    return False
                return self._a @ pts <= self._b
            elif pts.ndim == 2:
                if self.shape != pts.shape[1]:
                    return np.full(pts.shape[0], False, dtype=bool)
                return self._a[None, :, :] @ pts[:, :, None] <= self._b
            else:
                raise NotImplementedError

        def __contains_zonotope(other: Zonotope):
            # check all half-spaces bounding this given zonotope
            for i in range(self._a.shape[0]):
                b, _ = other.support_func(self._a[i].reshape((1, -1)), "u")
                if b > self._b[i]:
                    return False
            return True

        if isinstance(item, np.ndarray):
            return __contains_pts(item)
        elif isinstance(item, Geometry.Base):
            if item.type == Geometry.TYPE.INTERVAL:
                # TODO
                raise NotImplementedError
            elif item.type == Geometry.TYPE.POLYTOPE:
                # TODO
                raise NotImplementedError
            elif item.type == Geometry.TYPE.ZONOTOPE:
                return __contains_zonotope(item)

    def __str__(self):
        return self.info

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    # =============================================== class method
    @classmethod
    def functional(cls):
        raise NotImplementedError

    # =============================================== static method
    @staticmethod
    def empty(dim: int):
        raise NotImplementedError

    @staticmethod
    def rand(dim: int):
        num_vs = np.random.randint(5, 50)
        vs = np.random.rand(num_vs, dim)
        a, b = vertices_to_halfspaces(vs)
        return Polytope(a, b)

    # =============================================== public method
    def enclose(self, other):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

    def polygon(self, dims):
        """
        vertices of the projection onto the given 2 dimensions, in counterclockwise order
        """
        assert len(dims) == 2
        # the projection of a polytope is the convex hull of its projected vertices
        vs = self.vertices[:, dims]
        return vs[ConvexHull(vs).vertices, :]

    def proj(self, dims):
        vs = self.vertices[:, dims]
        if len(dims) == 1:
            return Polytope(np.array([[1.0], [-1.0]]), np.array([vs.max(), -vs.min()]))
        a, b = vertices_to_halfspaces(vs)
        return Polytope(a, b)

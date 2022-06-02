from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pypoman.polyhedron

import pyrat.util.functional.auxiliary as aux
from .geometry import Geometry

if TYPE_CHECKING:
    from .zonotope import Zonotope


class Polytope(Geometry.Base):
    def __init__(self, a: np.ndarray, b: np.ndarray):
        """
        aX<=b
        """
        assert not aux.is_empty(a)
        assert not aux.is_empty(b)
        assert a.ndim == 2 and a.shape[0] > 0
        assert b.ndim == 1 and a.shape[0] == b.shape[0]
        self._a = a
        self._b = b
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
            self._c = pypoman.polyhedron.compute_chebyshev_center(self._a, self._b)
        return self._c

    @property
    def dim(self) -> int:
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
            self._vs = np.stack(pypoman.compute_polytope_vertices(self._a, self._b))
        return self._vs

    @property
    def info(self):
        info = "\n ----------------- Polytope BEGIN -----------------\n"
        info += ">>> dimension -- constraints num\n"
        info += str(self.dim) + "\n"
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
                if self.dim != pts.shape[0]:
                    return False
                return self._a @ pts <= self._b
            elif pts.ndim == 2:
                if self.dim != pts.shape[1]:
                    return np.full(pts.shape[0], False, dtype=bool)
                return self._a[None, :, :] @ pts[:, :, None] <= self._b
            else:
                raise NotImplementedError

        def __contains_zonotope(other: Zonotope):
            # check all half spaces bounding this given zonotope
            for i in range(self._a.shape[0]):
                b = other.support_func(self._a[i].reshape((1, -1)), "u")
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
        raise NotImplementedError

    # =============================================== public method
    def enclose(self, other):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

    def polygon(self, dims):
        ineq = (self._a, self._b)
        e = np.zeros((2, self.dim))
        e[0, dims[0]] = 1
        e[1, dims[1]] = 1

        f = np.zeros(2)
        proj = (e, f)

        vs, _ = pypoman.projection.project_polyhedron(proj, ineq)
        vs = np.vstack(vs)
        center = np.sum(vs, axis=0) / vs.shape[0]

        angles = np.arctan2(vs[:, 1] - center[1], vs[:, 0] - center[0])
        angles[angles < 0] += 2 * np.pi
        idx = np.argsort(angles)
        return vs[idx]

    def proj(self, dims):
        raise NotImplementedError

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        from pyrat.util.functional import CSPSolver
        from pyrat.geometry import Interval, cvt2

        def f(x):
            z = Interval.zero(self._a.shape[0])
            for i in range(self.a.shape[0]):
                for j in range(self.a.shape[1]):
                    z[i] += self.a[i, j] * x[j]
                z[i] -= self._b[i]

            ind0 = np.logical_and(z.inf <= 0, z.sup >= 0)
            ind1 = z.inf > 0
            sum = np.sum(ind0)
            return 0 < sum and np.sum(ind1) <= 0

        lb = np.min(self.vertices, axis=0) - max_dist
        ub = np.max(self.vertices, axis=0) + max_dist
        boxes = CSPSolver.solve(f, lb, ub, max_dist)
        return [cvt2(box, element) for box in boxes]

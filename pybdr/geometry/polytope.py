from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pypoman.polyhedron

import pybdr.util.functional.auxiliary as aux
from scipy.spatial import ConvexHull
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
            self._c = pypoman.polyhedron.compute_chebyshev_center(self._a, self._b)
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
            self._vs = np.stack(pypoman.compute_polytope_vertices(self._a, self._b))
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
        a, b = pypoman.compute_polytope_halfspaces(vs)
        return Polytope(a, b)

    # =============================================== public method
    def enclose(self, other):
        raise NotImplementedError

    def reduce(self):
        raise NotImplementedError

    def polygon(self, dims):
        assert len(dims) == 2
        if self.shape == 2:
            vs = self.vertices
        else:

            ineq = (self._a, self._b)
            e = np.zeros((2, self.shape))
            e[[0, 1], dims] = 1

            f = np.zeros(2)
            proj = (e, f)

            vs = pypoman.projection.project_polytope(proj, ineq)
        hull = ConvexHull(vs)
        vs = np.asarray(vs)[hull.vertices, :]
        return vs

    def proj(self, dims):
        ineq = (self._a, self._b)
        e = np.zeros((len(dims), self.shape))
        e[np.arange(len(dims)), dims] = 1

        f = np.zeros(2)
        proj = (e, f)

        vs = pypoman.projection.project_polytope(proj, ineq)
        hull = ConvexHull(vs)
        vs = np.asarray(vs)[hull.vertices, :]
        a, b = pypoman.compute_polytope_halfspaces(vs)
        return Polytope(a, b)

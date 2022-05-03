from __future__ import annotations

import numpy as np
from numbers import Real
from numpy.typing import ArrayLike
import pyrat.util.functional.auxiliary as aux
from .geometry import Geometry
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zonotope import Zonotope


class IntervalTensor(Geometry.Base):
    def __init__(self, inf: ArrayLike, sup: ArrayLike):
        inf = inf if isinstance(inf, np.ndarray) else np.asarray(inf, dtype=float)
        sup = sup if isinstance(sup, np.ndarray) else np.asarray(sup, dtype=float)
        assert inf.shape == sup.shape
        assert np.all(inf <= sup)
        self._inf = inf
        self._sup = sup
        self._type = Geometry.TYPE.INTERVAL_TENSOR

        # =============================================== property

    @property
    def c(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        raise NotImplementedError

    @property
    def vertices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def info(self):
        raise NotImplementedError

    @property
    def type(self) -> Geometry.TYPE:
        raise NotImplementedError

    # =============================================== operator

    def __contains__(self, item):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

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

    def proj(self, dims):
        raise NotImplementedError

    def boundary(self, max_dist: float, element: Geometry.TYPE):
        # TODO
        raise NotImplementedError

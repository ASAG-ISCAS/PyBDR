from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
from pyrat.util.functional.aux_python import *
import numpy as np


@reg_classmethod
def empty(cls: Polyhedron, dim: int) -> Polyhedron:
    """
    construct empty set in R^n
    :param cls: instance of the polyhedron type
    :param dim: dimension of the target space
    :return: An empty set defined in R^n
    """
    p = cls._new(np.zeros((0, dim + 1), dtype=float), "h")
    p._int_empty = True
    p._int_fullspace = False
    p._int_lb = np.full((1, dim), np.inf, dtype=float)
    p._int_ub = np.full((dim, 1), -np.inf, dtype=float)
    return p

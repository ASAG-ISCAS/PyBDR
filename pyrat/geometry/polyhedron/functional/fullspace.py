from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
from pyrat.util.functional.aux_python import *
import numpy as np


@reg_classmethod
def fullspace(cls: Polyhedron, dim: int) -> Polyhedron:
    """
    construct the H-representation of R^n
    :param cls: instance of the Polyhedron type
    :param dim: dimension of the target space
    :return: A full-dimensional polyhedron
    """
    # R^n is represented by 0@x<=1
    h = np.zeros((1, dim + 1), dtype=float)
    h[:, -1] = 1
    p = cls._new(h, "h")
    p._irr_hrep = True
    p._int_empty = dim == 0  # R^0 is an empty set
    p._int_fulldim = dim > 0  # R^0 is not fully dimensional
    p._int_bounded = dim == 0  # R^0 is bounded
    p._int_lb = np.full((dim, 1), -np.inf, dtype=float)
    p._int_ub = np.full((dim, 1), np.inf, dtype=float)
    return p

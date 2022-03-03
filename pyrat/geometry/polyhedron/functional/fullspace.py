from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def fullspace(dim: int) -> Polyhedron:
    """
    construct the H-representation of R^n which represented by 0'@x<=1
    :param dim: dimension of the target space
    :return:
    """
    h = np.zeros((1, dim + 1), dtype=float)
    h[:, -1] = 1
    return Polyhedron(h)

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def empty(dim: int) -> "Polyhedron":
    """
    construct an empty set in R^n
    :param dim: dimension of the space
    :return:
    """
    return Polyhedron(np.zeros((0, dim + 1), dtype=float))

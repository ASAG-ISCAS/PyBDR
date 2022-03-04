from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
import numpy as np


def _mks_sub(lhs: Polyhedron, rhs: Polyhedron) -> Polyhedron:
    # TODO
    print("calling minkowski subtraction")
    h = np.zeros((3, 4), dtype=float)
    return Polyhedron(h)

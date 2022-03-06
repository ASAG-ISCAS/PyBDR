from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
import numpy as np
from pyrat.util.functional.aux_python import *


@reg_classmethod
def _mks_sub(cls: Polyhedron, lhs: Polyhedron, rhs: Polyhedron) -> Polyhedron:
    # TODO
    print("calling minkowski subtraction")
    h = np.zeros((3, 4), dtype=float)
    return cls._new(h, "h")

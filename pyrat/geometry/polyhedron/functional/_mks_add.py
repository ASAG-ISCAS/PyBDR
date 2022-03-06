from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from pyrat.util.functional.aux_python import *

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


@reg_classmethod
def _mks_add(cls: Polyhedron, lhs: Polyhedron, rhs: Polyhedron) -> Polyhedron:
    print("calling minkowski addition")
    # TODO
    h = np.zeros((3, 4), dtype=float)
    print(lhs._ieqh)
    return cls._new(h, "h")

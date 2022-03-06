from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron

from pyrat.util.functional.aux_python import *


@reg_property
def r(self: Polyhedron) -> np.ndarray:
    if not self.has_vrep and self.has_hrep:
        self.compute_vrep()
    return self._r

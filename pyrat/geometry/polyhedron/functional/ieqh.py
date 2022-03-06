from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron

from pyrat.util.functional.aux_python import *


@reg_property
def ieqh(self: Polyhedron) -> np.ndarray:
    if not self.has_hrep and self.has_vrep:
        self.compute_hrep()
    return self._ieqh

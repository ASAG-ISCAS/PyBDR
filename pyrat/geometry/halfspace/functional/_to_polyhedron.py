from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace
from pyrat.geometry import Polyhedron


def _to_polyhedron(self: HalfSpace) -> Polyhedron:
    return Polyhedron(np.concatenate([self.c, self.d], axis=0))

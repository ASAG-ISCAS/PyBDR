from __future__ import annotations
from pyrat.util.functional.aux import *
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def _init_from_vrep(self: Polyhedron, vrep: np.ndarray):
    """
    init polyhedron from vertices as matrix storing column vectors
    :param self: calling polyhedron
    :param vrep: vertices
    :return:
    """
    # replace nearly-zero entries by zero
    vrep[abs(vrep) < self._zo_tol] = 0
    self._dim = vrep.shape[0]
    self._v = vrep
    self._ieqh = np.zeros((0, self._dim + 1), dtype=float)
    self._eqh = np.zeros((0, self._dim + 1), dtype=float)
    self._r = np.zeros((0, self._dim), dtype=float)
    self._has_hrep = not is_empty(self._ieqh) or not is_empty(self._eqh)
    self._has_vrep = not is_empty(self._v) or not is_empty(self._r)
    # compute minimum representation for the affine set
    if not is_empty(self._eqh):
        if np.linalg.norm(self._eqh, ord=2) == 0 and (
            is_empty(self._ieqh) or np.linalg.norm(self._ieqh, ord=2) == 0
        ):
            # corner case 0@x=0
            h = np.zeros((1, self._dim + 1), dtype=float)
            h[:, -1] = 1
            self._init_from_hrep(h)
            self._irr_hrep = True  # full space representation
        else:
            self._eqh = min_affine(self._eqh)

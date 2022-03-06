from __future__ import annotations
from typing import TYPE_CHECKING
from pyrat.util.functional.aux_numpy import *

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def _normalize_hrep(hrep: np.ndarray, zo_tol: float) -> np.ndarray:
    h = hrep
    # 2-norm of each facet
    norm = np.linalg.norm(hrep[:, :-1], ord=2, axis=1)
    # normalize 0'@x<=+/-b to 0'@x<=+/- sign(b)
    zo_rows = norm < zo_tol
    norm[zo_rows] = 1
    h[:, -1][zo_rows] = np.sign(hrep[:, -1])
    # normalize each halfspace (0@x<=b will be left intact)
    h /= norm[:, None]
    return h


def normalize(self: Polyhedron):
    if self._has_hrep:
        self._ieqh = _normalize_hrep(self._ieqh, self._zo_tol)
        # normalize equalities if present
        if not is_empty(self._eqh):
            self._eqh = _normalize_hrep(self._eqh, self._zo_tol)

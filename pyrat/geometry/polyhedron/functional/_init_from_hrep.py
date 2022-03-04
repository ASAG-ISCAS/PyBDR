from __future__ import annotations
from pyrat.util.functional.aux import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def _init_from_hrep(self: Polyhedron, hrep: np.ndarray):
    """
    init polyhedron from hrep as linear inequalities
    :param self: calling polyhedron
    :param arr: linear inequalities hrep=[m,v] indicates m@x<=v
    :return:
    """
    # assert hrep.shape[0] == hrep.shape[1] - 1
    # replace c'@x<=+/-inf by 0'@x<=+/-1
    inf_rows = np.isinf(hrep[:, -1])
    if any(inf_rows):
        hrep[inf_rows, :-1] = 0
        hrep[inf_rows, -1] = np.sign(hrep[:, -1])
    # replace nearly-zero entries by zero
    hrep[abs(hrep) < self._zo_tol] = 0
    self._dim = hrep.shape[0]
    self._ieqh = hrep
    self._eqh = np.zeros((0, self._dim + 1), dtype=float)
    self._v = np.zeros((0, self._dim), dtype=float)
    self._r = np.zeros((0, self._dim), dtype=float)
    self._has_hrep = not is_empty(self._ieqh)

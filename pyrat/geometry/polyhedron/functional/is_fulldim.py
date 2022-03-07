from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
from pyrat.util.functional.aux_numpy import *
from pyrat.util.functional.aux_python import *


@reg_property
def is_fulldim(self: Polyhedron) -> bool:
    """
    check if this polyhedron is full-dimensional or not
    :param self: this polyhedron instance
    :return: TRUE if full-dimensional, FALSE otherwise
    """
    if self._int_fulldim is None:
        # if the polyhedron is empty -> not full dimensional
        if self.is_empty:
            self._int_fulldim = False
        else:
            # compute interior point only if self.is_empty property has not been set
            sol = self._int_inner_pt
            self._int_fulldim = sol.is_strict and not is_empty(sol.x)
    return self._int_fulldim

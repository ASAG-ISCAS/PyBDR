from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def compute_vrep(self: Polyhedron):
    """
    H to V conversion with possible redundant V-rep output
    :param self: this polyhedron instance
    :return:
    """
    if self.has_vrep:
        return  # do nothing
    elif not self.has_vrep:
        # empty set
        self._has_vrep = True
        return
    elif self.is_empty:
        # empty set = empty vertices and rays
        self._v = np.zeros((0, self.dim), dtype=float)
        self._r = np.zeros((0, self.dim), dtype=float)
        self._has_vrep = True
        return
    done, backup_tried = False, False
    # work with minimal H-representations to improve numerics
    self._compute_min_hrep()
    # shift the polytope such that it contains the origin in its interior
    xc = np.zeros((self.dim, 1), dtype=float)
    if self.is_bounded:
        # TODO
        pass

    # TODO
    pass

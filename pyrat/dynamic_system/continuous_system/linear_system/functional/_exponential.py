from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.dynamic_system import LinearSystem

from . import Options
import numpy as np


def _exponential(self: LinearSystem, p: Options):
    """
    compute the over-approximation of the exponential of a system matrix up to a certain
    accuracy
    :param self: this Linear system instance
    :param p: options for the computation
    :return:

    refs:

    [1] Althoff, M., Le Guernic, C., & Krogh, B. H. (2011, April). Reachable set
    computation for uncertain time-varying linear systems. In Proceedings of the 14th
    international conference on Hybrid systems: computation and control (pp. 93-102).
    """
    e = np.eye(self.dim, dtype=float)
    xa_power = [self._xa]
    xa_power_abs = [abs(self._xa)]
    # compute powers for each term and sum of these
    for i in range(p.taylor_terms):
        # compute powers
        xa_power.append(xa_power[-1] @ self._xa)
        # TODO

        pass

    pass

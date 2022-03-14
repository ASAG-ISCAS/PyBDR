from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.dynamic_system import LinearSystem

from . import Options


def init_reach_euclidean(self: LinearSystem, r0, p: Options):
    """
    compute the reachable continuous set for the first time step in the untransformed space
    :param self: this Linear system instance
    :param r0: initial reachable set
    :param p: options for the computation
    :return:
    """
    # compute exponential matrix


    pass

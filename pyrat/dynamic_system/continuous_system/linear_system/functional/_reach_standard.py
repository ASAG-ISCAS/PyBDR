from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.special

if TYPE_CHECKING:
    from pyrat.dynamic_system import LinearSystem

from . import Options
import scipy.special as sp


def _reach_standard(self: LinearSystem, p: Options):
    """
    compute the reachable set for linear systems using the standard (non-wrapping-free)
    reachability algorithm for linear systems
    :param self: this Linear system instance
    :param p: options for the computation
    :return: resulting reachable set

    refs:
    [1] Girard, A. (2005, March). Reachability of uncertain linear systems using
    zonotopes. In International Workshop on Hybrid Systems: Computation and Control
    (pp. 291-305). Springer, Berlin, Heidelberg.
    """
    # obtain factors for initial state and input solution
    idx = np.arange(p.taylor_terms)
    p.factors = np.power(p.time_step, idx) / sp.factorial(idx)
    # if a trajectory should be tracked
    if p.u_track_vec is None:
        input_corr = 0
    # initialize reachable set computation

    raise NotImplementedError
    # TODO

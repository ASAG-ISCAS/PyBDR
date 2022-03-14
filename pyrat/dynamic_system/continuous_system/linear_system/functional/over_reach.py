from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.dynamic_system import LinearSystem

from . import Options


def over_reach(self: LinearSystem, p: Options):
    """
    compute tight over approximation reachable set
    :param self:
    :param p: parameters
    :return:
    """
    if p.algo == "std":
        return self._reach_standard(p)
    elif p.algo == "adp":
        return self._reach_adaptive(p)
    elif p.algo == "wf":
        return self._reach_wrapping_free(p)
    elif p.algo == "decmp":
        return self._reach_decomp(p)
    elif p.algo == "fs":
        return self._reach_from_start(p)
    elif p.algo == "kl":
        return self._reach_krylov(p)
    else:
        raise NotImplementedError

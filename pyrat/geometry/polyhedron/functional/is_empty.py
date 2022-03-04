from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
from pyrat.util.functional.aux import *


def is_empty(self: Polyhedron) -> bool:
    """
    check if this polyhedron is empty or not
    empty region is considered if it is not feasible and if the diameter of the
    chebyshev ball inscribed inside the polyhedron is less than "region_tol"
    :param self: this polyhedron instance
    :return:
    """
    # compute interior point only if P(i). _int_empty property has not been set
    if self.dim == 0 or (
        is_empty(self._ieqh)
        and is_empty(self._eqh)
        and is_empty(self._v)
        and is_empty(self._r)
    ):
        self._int_empty = True
    elif self._int_empty is None:
        # note that low-dimensional polyhedra have radius=0
        # we need to check hwo far is the interior point from the inequalities
        ip = self._int_inner_pt
        if not is_empty(self._r) and 2 * self._int_inner_pt.r < self._region_tol:
            if self._int_inner_pt.is_strict:
                self._int_empty = True
            else:
                # low-dimensional
                if self.has_vrep:
                    # calculate the maximum distance to vertices
                    hn = self._v - np.tile(ip.x, (2, 1))
                    d = np.sqrt(np.sum(hn**2, axis=1))
                    dmax = np.max(d)
                else:
                    # calculate teh maximum violation of inequalities
                    hn = self.ieqh @ np.concatenate([ip.x, -1], axis=1)
                    dmax = np.max(hn)
                if dmax > self._abs_tol:
                    # maximum allowable tolerance is violated, region is empty
                    self._int_empty = True
                else:
                    self._int_empty = False
        elif np.any(np.isnan(ip.x)):
            # NaN in interior point means empty polyhedron
            self._int_empty = True
        else:
            self._int_empty = is_empty(ip.x)
    return self._int_empty

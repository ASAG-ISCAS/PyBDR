from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron

from pyrat.util.functional.aux_numpy import *
import scipy.spatial as sp


def compute_hrep(self: Polyhedron):
    """
    V to H conversion with possible redundant H-rep output
    :param self: this polyhedron instance
    :return:
    """
    if self.has_hrep:
        return  # do nothing
    elif not self.has_vrep:
        # empty set
        self._has_hrep = True
        return
    elif self.is_fullspace:
        # R^n
        rn = self.__class__.fullspace(self.dim)
        self._ieqh = rn.ieqh
        self._eqh = rn.eqh
        self._has_hrep = True
        return

    # compute H-rep
    a, b, lin = None, None, None
    self._compute_min_vrep()
    if (
        is_empty(self.r)
        and self.dim > 1
        and self.is_fulldim
        and self.v.shape[0] >= self.dim + 1
        and self.v.shape[1] <= 3
    ):
        """
        try to compute convex hull first; requires following conditions to be met:
            + no rays
            + dimension at least 2
            + the set is full-dimensional
            + the set has at least d+1 vertices
        """
        x0 = self._int_inner_pt.x
        v = self.v
        k = sp.ConvexHull(v).vertices
        d = v.shape[1]
        a = np.zeros((k.shape[0], d), dtype=float)
        b = np.ones((k.shape[0], 1), dtype=float)
        for i in range(k.shape[0]):
            # each row of K contains indices of vertices that lie on the i-th facet
            p = v[v[i, :], :]
            # compute the normal vector and the offset of the facet
            w = np.concatenate([p, -np.ones((d, 1), dtype=float)], axis=0)
            q, _ = np.linalg.qr(w.T)
            a_ = a[:d, -1]
            b_ = a[-1, -1]
            # determine the sign
            if a @ x0 > b:
                a = -a_
                b = -b_
            a[i, :] = a_.T
            b[i] = b_

    else:
        # do facet enumeration with CDD->using pycddlib or using scipy.spatial.ConvexHull
        raise NotImplementedError
        # TODO

    h = np.concatenate([a, b], axis=0)
    ieqh = np.delete(h, lin, axis=0)
    eqh = h[lin, :]
    self._ieqh = ieqh
    self._eqh = eqh
    self._has_hrep = True

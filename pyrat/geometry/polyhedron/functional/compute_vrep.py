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
        xc = self._int_cheby_data[0]
    ieqa = self.ieqa
    ieqb = self.ieqb - ieqa @ xc
    eqa = self.eqa
    eqb = self.eqb - eqa @ xc
    a = np.concatenate([eqa, ieqa], axis=0)  # equalities must come first
    b = np.concatenate([eqb, ieqb], axis=0)
    # do vertex enumeration
    while not done:
        try:
            # change almost-zeero elements to zero
            a[abs(a) < self._zo_tol] = 0
            b[abs(b) < self._zo_tol] = 0
            # round H-representation to certain number of decimal places
            a = np.around(a, decimals=15)
            b = np.around(b, decimals=15)
            lin = np.arange(self.eqh.shape[0])
            ret_v, ret_r, ret_a, ret_b = self.lrs("extreme")  # not implemented yet
            # shift vertices back
            ret_v += np.tile(xc.T, (ret_v.shape[0], 1))
            if ret_v.shape[0] == 0:  # this is a cone ... we need an explicit vertex
                self._v = np.concatenate(
                    [self._v, np.zeros((1, self.dim), dtype=float)], axis=0
                )
            else:
                self._v = np.concatenate([self._v, ret_v], axis=0)
            self._r = np.concatenate([self._r, ret_r], axis=0)
            done = True
        except:
            if backup_tried:
                self._v = np.concatenate(
                    [self._v, np.zeros((1, self.dim), dtype=float)], axis=0
                )
                done = True
                """
                this error appears usually when plotting polyhedra, it is therefore 
                disabled to show at least something error like:
                Could not compute vertices: Numerical problems in CDD
                """
            backup_tried = True
            """
            use trick from mpt2.6 and reduce the calculation precision CDD sometimes fails 
            to compute extreme points correctly, this happens often when slopes of two 
            hyperplanes are too close, that's why we use a fixed-point arithmetics
            """
            raise NotImplementedError
            # TODO

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry.zonotope import Zonotope
from pyrat.geometry.polyhedron import Polyhedron
from pyrat.util.functional.aux_python import *
from pyrat.util.functional.aux_numpy import *
from math import factorial


"""
Refs:
    [1] M. Althoff, "On Computing the Minkowski Difference of Zonotopes"
"""


@reg_classmethod
def _approx_mink_diff_althoff(cls: Zonotope, lhs: Zonotope, rhs: Zonotope) -> Zonotope:
    p, comb, _ = lhs.to("polyhedron")
    h, k = p.ieqa, p.ieqb
    h_ori = h[: 0.5 * h.shape[0]]
    # number of rhs generators
    rhs_gen_num = rhs.gen_num
    # intersect polytopes according to Theorem 3 in [1]
    delta_k = h @ rhs.z[:, 1]
    for i in range(1, rhs_gen_num):
        delta_k = delta_k + abs(h @ rhs.z[:, i + 1])
    k_new = k - delta_k
    p_int = Polyhedron(np.concatenate([h, k_new], axis=1))
    # remove redundant half-spaces and remember indices
    rm_ind = p_int.removed_halfspaces(h_ori)
    # is the Minkowski difference an empty set?
    if rm_ind is None:
        # Minkowski difference is an empty set
        return cls.empty(0)  # empty set in R^0
    else:
        """
        Minkowski difference is not empty, but not all generators of the minuend are
        required
        """
        if not is_empty(rm_ind):
            # count generators that have been removed
            # number of original generators
            gens_num = lhs.gen_num
            # initialize indices
            indices = np.zeros((1, gens_num))
            # add indices of generators that contributed to the removed halfspace
            for i in range(rm_ind.shape[0]):
                contributing_indices = comb[rm_ind[i], :]
                indices[contributing_indices] += 1
            # find generators to be removed
            n = lhs.gen_num
            required_removals = factorial(gens_num - 1) / (
                factorial(n - 2) * factorial(gens_num - n + 1)
            )
            ind_remove = indices == required_removals
            # obtain reduced minuend
            g = lhs.generator
            g = np.delete(g, ind_remove)
            lhs._z = np.concatenate([lhs.center, g], axis=1)
            # remove h,k
            c = h_ori
            c = np.delete(c, ind_remove)
            d = k_new[: 0.5 * k_new.shape[0], :]
            d = np.delete(d, ind_remove)
        else:
            c = h_ori
            d = k_new[: 0.5 * k_new.shape[0], :]
        # compute center
        c = lhs.center - rhs.center
        # obtain minuend generators
        g = lhs.generator
        # reverse computation from halfspace generation
        delta_d = d = c @ lhs.center + c @ rhs.center
        a_abs = abs(c @ g)
        alpha = np.linalg.pinv(a_abs) @ delta_d  # solve linear set of equations
        g_new = np.zeros((lhs.dim, alpha.shape[0]), dtype=float)
        for i in range(alpha.shape[0]):
            g_new[:, i] = alpha[i] * lhs.z[:, i + 1]
        return cls._new(np.concatenate([c, g_new], axis=1))

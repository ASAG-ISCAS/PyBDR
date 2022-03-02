from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry.zonotope import Zonotope
from .cvt2polyhedron import cvt2polyhedron
from pyrat.geometry.polyhedron import Polyhedron

"""
Refs:
    [1] M. Althoff, "On Computing the Minkowski Difference of Zonotopes"
"""


def approx_mink_diff_althoff(lhs: Zonotope, rhs: Zonotope) -> Zonotope:
    p, comb, _ = cvt2polyhedron(lhs)
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

    # TODO
    raise Exception("NOT SUPPORTED YET")

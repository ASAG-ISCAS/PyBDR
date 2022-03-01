from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry.zonotope import Zonotope
from .cvt2polyhedron import cvt2polyhedron


def approx_mink_diff_althoff(lhs: Zonotope, rhs: Zonotope) -> Zonotope:
    p, comb_g, is_full_dim = cvt2polyhedron(lhs)
    # TODO
    raise Exception("NOT SUPPORTED YET")

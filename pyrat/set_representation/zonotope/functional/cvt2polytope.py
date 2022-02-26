from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.set_representation.zonotope import Zonotope
    from pyrat.set_representation.polytope import Polytope


def cvt2polytope(lhs: Zonotope) -> Polytope:
    # TODO
    raise Exception("not supported yet")

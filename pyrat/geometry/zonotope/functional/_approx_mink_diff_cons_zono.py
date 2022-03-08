from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry.zonotope import Zonotope
from pyrat.util.functional.aux_python import *


@reg_classmethod
def _approx_mink_diff_cons_zono(
    cls: Zonotope, lhs: Zonotope, rhs: Zonotope
) -> Zonotope:
    # TODO
    raise Exception("NOT SUPPORTED YET")

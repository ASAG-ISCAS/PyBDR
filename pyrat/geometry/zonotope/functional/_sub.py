from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope
from pyrat.util.functional.aux_python import *


def __sub__(
    self: Zonotope, other: Zonotope | numbers.Real, method: str = "althoff"
) -> Zonotope:
    if isinstance(other, numbers.Real):
        z = self._z.copy()
        z[:, :1] -= other
        return self._new(self.__class__, z)
    try:
        if method == "althoff":
            return self._approx_mink_diff_althoff(self, other)
        elif method == "cons_zono":
            return self._approx_mink_diff_cons_zono(self, other)
        else:
            raise Exception("Unsupported method")
    except:
        raise Exception("Invalid operand to do Minkowski difference")


def __isub__(
    self: Zonotope, other: Zonotope | numbers.Real, method: str = "althoff"
) -> Zonotope:
    return self.__sub__(other, method)


def __rsub__(
    self: Zonotope, other: Zonotope | numbers.Real, method: str = "althoff"
) -> Zonotope:
    raise NotImplementedError
    # TODO

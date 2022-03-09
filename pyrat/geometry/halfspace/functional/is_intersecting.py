from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def is_intersecting(self: HalfSpace, other, opt: str = "exact") -> bool:
    """
    check if this halfspace is intersecting with other
    :param self:
    :param other:
    :param opt:
    :return:
    """
    raise NotImplementedError
    # TODO

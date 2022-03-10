from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def contains(self: Zonotope, other) -> bool:
    """
    check if other geometry object enclosed by this zonotope
    :param self: this zonotope instance
    :param other: other geometry object
    :return:
    """
    raise NotImplementedError
    # TODO

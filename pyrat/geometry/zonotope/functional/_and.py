from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def __and__(self: Zonotope, other: Zonotope, method: str = "con_zono") -> Zonotope:
    """
    compute intersection with other zonotope
    :param self: this zonotope instance
    :param other: other geometry object
    :param method:
    :return: resulting zonotope indicates the intersection
    """
    # quick check: simpler function for intervals
    if self.is_interval and other.is_interval:
        # conversion to intervals exact
        return (self.to("interval") & other.to("interval")).to("zonotope")
    raise NotImplementedError
    # TODO

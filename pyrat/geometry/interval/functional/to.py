from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Interval


def to(self: Interval, target: str):
    if target == "zonotope":
        return self._to_zonotope()
    raise NotImplementedError
    # TODO

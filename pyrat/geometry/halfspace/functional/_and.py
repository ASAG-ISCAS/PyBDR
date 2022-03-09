from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def __and__(self: HalfSpace, other):
    raise NotImplementedError

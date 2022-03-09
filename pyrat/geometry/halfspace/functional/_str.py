from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace


def __str__(self: HalfSpace) -> str:
    return str(self.c) + " " + str(self.d)

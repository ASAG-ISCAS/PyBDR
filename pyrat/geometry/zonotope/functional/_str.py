from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope


def __str__(self: Zonotope) -> str:
    return "center: " + str(self.center) + "\ngenerator: \n" + str(self.generator)

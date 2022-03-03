from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import TaylorModel


def demo_func(self: TaylorModel):
    print(self._name)
    self._index = 2

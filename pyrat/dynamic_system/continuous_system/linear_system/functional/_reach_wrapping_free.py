from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.dynamic_system import LinearSystem

from . import Options


def _reach_wrapping_free(self: LinearSystem, p: Options):
    raise NotImplementedError
    # TODO

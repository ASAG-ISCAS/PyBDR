from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.dynamic_system import LinearSystem

from pyrat.util.functional.aux_python import *


@reg_property
def dim(self: LinearSystem) -> int:
    raise NotImplementedError
    # TODO

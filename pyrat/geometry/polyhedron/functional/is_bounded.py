from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron

from pyrat.util.functional.aux_python import *


@reg_property
def is_bounded(self: Polyhedron) -> bool:
    raise NotImplementedError
    # TODO

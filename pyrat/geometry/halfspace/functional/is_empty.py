from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace

from pyrat.util.functional.aux_python import *
import pyrat.util.functional.aux_numpy as an


@reg_property
def is_empty(self: HalfSpace) -> bool:
    """
    check if this halfspace is empty or not
    :param self: this halfspace instance
    :return: TRUE if this halfspace is empty
    """
    return an.is_empty(self.c) and isinstance(self.d, numbers.Real)

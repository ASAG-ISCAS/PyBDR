from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *


@reg_property
def is_interval(self: Zonotope) -> bool:
    """
    check if a zonotope can be equivalently represented by an interval object (all
    generator axis-aligned)
    :param self: this zonotope instance
    :return: TRUE if this zonotope represents an interval, otherwise FALSE
    """
    if self.dim == 1:
        return True
    nz_num = np.count_nonzero(self.generator, axis=1)
    return False if np.any(nz_num > 1) else True

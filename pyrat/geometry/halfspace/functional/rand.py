from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import HalfSpace

from pyrat.util.functional.aux_python import *


@reg_classmethod
def rand(cls: HalfSpace, dim: int = None) -> HalfSpace:
    """
    generate a random halfspace
    :param cls: this halfspace type instance
    :param dim: dimension of the target space
    :return: resulting halfspace in target dimension
    """

    dim = np.random.randint(0, 10) if dim is None else dim
    # ranges
    lb, ub = -10, 10
    # instantiate interval
    c = lb + np.random.rand(dim) * (ub - lb)
    d = lb + np.random.rand(1) * (ub - lb)
    return cls._new(c, d)

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Zonotope

from pyrat.util.functional.aux_python import *


@reg_classmethod
def rand_fix_dim(cls: Zonotope, dim: int) -> Zonotope:
    assert dim > 0
    gen_num = np.random.randint(0, 10)
    return cls._new(np.random.rand(dim, gen_num))

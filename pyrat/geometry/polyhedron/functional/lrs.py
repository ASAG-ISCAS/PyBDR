from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron


def lrs(self: Polyhedron, method: str):
    """
    implementation of the LRS algorithm

    enumerates extreme points or extreme facets of a polytope, works only with full
    dimensional polyhedra

    :param self:
    :param method:
    :return:
    """
    ret_v, ret_r, ret_h, ret_k = None, None, None, None
    raise NotImplementedError
    # TODO

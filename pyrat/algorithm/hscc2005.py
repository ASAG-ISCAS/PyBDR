"""
REF:

Girard, A. (2005, March). Reachability of uncertain linear systems using zonotopes. In
International Workshop on Hybrid Systems: Computation and Control (pp. 291-305).
Springer, Berlin, Heidelberg.
"""

from __future__ import annotations
import numpy as np
from numbers import Real
from dataclasses import dataclass
from pyrat.geometry import Geometry
from pyrat.dynamic_system import LinSys
from .algorithm import Algorithm
from scipy.linalg import expm


class HSCC2005:
    @dataclass
    class Options(Algorithm.Options):
        # TODO

        def validation(self, dim: int):
            # TODO
            return True

    def one_step(self, sys: LinSys, r0, opt: Options):

        raise NotImplementedError

    def reach(self, sys: LinSys, opt: Options):
        assert opt.validation(sys.dim)
        # TODO
        raise NotImplementedError

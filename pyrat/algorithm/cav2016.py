"""
REF:

Xue, B., She, Z., & Easwaran, A. (2016, July). Under-approximating backward reachable
sets by polytopes. In International Conference on Computer Aided Verification
(pp. 457-476). Springer, Cham.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry
from pyrat.geometry.operation import cvt2, boundary
from pyrat.misc import Reachable, Set
from .algorithm import Algorithm
from .cdc2008 import CDC2008


class CAV2016:
    @dataclass
    class Options(Algorithm.Options):
        epsilon_m: float = np.inf  # for boundary sampling
        epsilon: float = np.inf  # for backward verification

        def validation(self, dim: int):
            assert self._validate_time_related()
            assert not np.isinf(self.epsilon) and self.epsilon >= 0
            assert not np.isinf(self.epsilon_m) and self.epsilon_m > 0
            self.step_idx = self.steps_num - 1  # index from 0
            return True

    @classmethod
    def boundary_back(cls, sys: NonLinSys.Entity, u, epsilon, opt: CDC2008.Options):
        bounds = boundary(u, epsilon, Geometry.TYPE.ZONOTOPE)
        r0 = [Set(cvt2(bd, Geometry.TYPE.ZONOTOPE)) for bd in bounds]
        r_ti, r_tp = CDC2008.reach_one_step(sys, r0, opt)

        # TODO
        raise NotImplementedError

    @classmethod
    def polytope(cls, omega):
        # TODO
        raise NotImplementedError

    @classmethod
    def contraction(cls, omega, o, opt: Options):
        # TODO
        raise NotImplementedError

    @classmethod
    def verification(cls, pre_u, cur_u):
        # TODO
        return True

    @classmethod
    def one_step_backward(cls, u, sys, opt: Options, opt_back: CDC2008.Options):
        omega = cls.boundary_back(sys, u, opt.epsilon_m, opt_back)
        o = cls.polytope(omega)
        cur_u = cls.contraction(omega, o, opt)
        if not cls.verification(u, cur_u):
            return None, False
        return cur_u, True

    @classmethod
    def reach(cls, sys: NonLinSys.Entity, opt: Options, opt_back: CDC2008.Options):
        assert opt.validation(sys.dim)
        assert opt_back.validation(sys.dim)
        ti_set, ti_time = [], []
        time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
        u = opt.r0
        ti_set.append(u)
        ti_time.append(time_pts[-1])

        # loop over all backward steps
        while opt.step_idx >= 0:
            u, is_valid = cls.one_step_backward(u, sys, opt, opt_back)
            opt.step_idx -= 1
            if is_valid:
                ti_set.append(u)
                ti_time.append(time_pts[-opt.step_idx - 1])
            else:
                break

        return Reachable.Result(ti_set, ti_time)

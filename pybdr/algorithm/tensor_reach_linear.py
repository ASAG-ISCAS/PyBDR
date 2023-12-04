from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pybdr.dynamic_system import LinSys
from pybdr.geometry import Interval
from .algorithm import Algorithm
from scipy.linalg import expm


class TensorReachLinear:
    @dataclass
    class Options(Algorithm.Options):
        taylor_terms: int = 4
        taylor_powers = None
        taylor_err = None

        def validation(self, dim: int):
            # TODO
            return True

    @staticmethod
    def exponential(sys: LinSys, opt: Options):
        pass

    @classmethod
    def compute_time_interval_err(cls, sys: LinSys, opt: Options):
        pass

    @classmethod
    def input_time_interval_err(cls, sys: LinSys, opt: Options):
        pass

    @classmethod
    def input_solution(cls, sys: LinSys, opt: Options):
        pass

    @classmethod
    def err_solution(cls, v_dyn: Interval, opt: Options):
        pass

    @classmethod
    def delta_reach(cls, sys: LinSys, r: Interval, opt: Options):
        pass

    @classmethod
    def reach_one_step(cls, sys: LinSys, r: Interval, opt: Options):
        cls.exponential(sys, opt)
        cls.compute_time_interval_err(sys, opt)
        cls.input_solution(sys, opt)
        opt.taylor_ea_t = expm(sys.xa * opt.step)
        
        pass

    @classmethod
    def reach(cls, sys: LinSys, opt: Options):
        assert opt.validation(sys.dim)
        # init container for storing teh results
        time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
        ti_set, ti_time, tp_set, tp_time = [], [], [opt.r0], [time_pts[0]]

        while opt.step_idx < opt.steps_num - 1:
            # TODO
            pass

        return ti_set, tp_set, np.vstack(ti_time), np.array(tp_time)

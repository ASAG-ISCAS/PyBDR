from __future__ import annotations
import numpy as np
from dataclasses import dataclass

import scipy.special
from pyrat.util.result_manager import ReachableResult, SimulationResult


class NonLinearSystem:
    @dataclass(init=True)
    class Parameters:
        time_start: float = 0
        time_end: float = 100
        steps: int = 10
        r0 = None
        ru = None
        spec = None

    @dataclass
    class Options:
        time_step: float = None
        taylor_terms: int = 1
        zonotope_order: int = 50
        algo: str = "lin"
        tensor_order: int = 2
        lagrange_rem = {}
        factors: np.ndarray = None
        u_trans_vec = None
        u_trans = None
        t: float = 0

        def validate(self, param: NonLinearSystem.Parameters):
            # TODO do nothing now
            raise NotImplementedError

    class System:
        def __init__(self, model):
            self._model = model

        # =============================================== property
        @property
        def name(self):
            return self._model.name

        @property
        def model(self):
            return self._model

        # =============================================== operators
        def __str__(self):
            raise NotImplementedError
            # TODO

        # =============================================== private method
        def _init_reach_over(self, r0, opt: NonLinearSystem.Options):
            raise NotImplementedError
            # TODO

        def _reach_over_standard(
            self, p: NonLinearSystem.Parameters, opt: NonLinearSystem.Options
        ):
            # obtain factors for initial state and input solution time step
            i = np.arange(opt.taylor_terms)
            opt.factors = np.power(opt.time_step, i) / scipy.special.factorial(i)
            # if a trajectory should be tracked
            if opt.u_trans_vec is not None:
                opt.u_trans = opt.u_trans_vec[:, 0]
            # set start time point
            opt.t = p.time_start
            # time period
            ts = np.linspace(p.time_start, p.time_end, p.steps, endpoint=True)
            # init resulting reachable set
            r = ReachableResult(p.steps, p.steps)
            # init reachable set computation
            r_next, opt = self._init_reach_over(p.r0, opt)
            # loop over all steps
            for i in range(1, opt.steps):
                # TODO
                raise NotImplementedError

            return r

        def _reach_over_adaptive(
            self, param: NonLinearSystem.Parameters, opt: NonLinearSystem.Options
        ):
            raise NotImplementedError
            # TODO

        # =============================================== public method
        def reach_over(
            self, param: NonLinearSystem.Parameters, opt: NonLinearSystem.Options
        ):
            opt.validate(param)  # validate given options
            if opt.algo == "standard":
                return self._reach_over_standard(param, opt)
            elif opt.algo == "adaptive":
                return self._reach_over_adaptive(param, opt)
            else:
                raise NotImplementedError

        def reach_under(
            self, param: NonLinearSystem.Parameters, opt: NonLinearSystem.Options
        ):
            raise NotImplementedError

        def simulate_rand(
            self, param: NonLinearSystem.Parameters, opt: NonLinearSystem.Options
        ):
            raise NotImplementedError

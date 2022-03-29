from __future__ import annotations

import copy

import numpy as np
from dataclasses import dataclass
import sympy as sy
import scipy.special
from .linear_system import LinearSystem
from pyrat.misc import ReachableResult, SimulationResult


class NonLinearSystem:
    @dataclass(init=True)
    class Parameters:
        time_start: float = 0
        time_end: float = 100
        steps: int = 10
        r0: list = None
        ru = None
        spec = None

    @dataclass
    class Options:
        time_step: float = None
        t_start: float = None
        taylor_terms: int = 1
        zonotope_order: int = 50
        algo: str = "lin"
        tensor_order: int = 2
        lagrange_rem = {}
        factors: np.ndarray = None
        u_trans_vec = None
        u_trans = None
        u = None
        t: float = 0

        def validate(self, param: NonLinearSystem.Parameters):
            # TODO do nothing now
            raise NotImplementedError

    class System:
        def __init__(self, model):
            self._model = model
            self._lin_error = {}

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
        def jacobian(self, x, u):
            # TODO
            raise NotImplementedError

        def evaluation(self, x, u, mode: str = "numpy"):
            f = sy.lambdify(self._model.variables, self._model.f, mode)
            return f(x, u)

        def linearize(
            self, opt: NonLinearSystem.Options, r
        ) -> (LinearSystem.System, LinearSystem.Options):
            """
            linearize the nonlinear system, linearization error is n;ot included yet
            :param opt: options for the linearization
            :param r: actual reachable set
            :return:
            """
            # linearization point p.u of the input is the center of the input set
            p = {"u": opt.u_trans}
            # obtain linearization point
            if hasattr(opt, "lin_pt"):
                p["x"] = opt.lin_pt
            elif hasattr(opt, "ref_pts"):
                cur_step = int(np.ceil((opt.t - opt.t_start) / opt.time_step))
                p["x"] = 1 / 2 * np.sum(opt.ref_pts[:, cur_step : cur_step + 1], axis=1)
            else:
                # linearization point p.x of the state is the center of the last
                # reachable set R translated by 0.5*delta_t*f0
                f0_prev = self.evaluation(r.center, p["u"])
                try:  # if time step not yet created
                    p["x"] = r.center + f0_prev * 0.5 * opt.time_step
                except:
                    print("time step does NOT created yet")
                    p["x"] = r.center
            # substitute p into the system equation to obtain the constraint input
            f0 = self.evaluation(p["x"], p["u"])
            # substitute p into the Jacobin with respect to x and u to obtain the
            # system matrix A and teh input matrix B
            a, b = self.jacobian(p["x"], p["u"])
            a_lin, b_lin = a, b
            opt_lin = opt
            if opt_lin.algo == "lin_rem":
                # in order to compute da, db, we use the reachability set computed for
                # one step in init_reach
                raise NotImplementedError
            else:
                # set up linearized system
                lin_sys = LinearSystem.System(a, np.ones(len(self._model.variables[1])))
            # set up options for linearized system
            opt_lin.u = b * (opt.u + opt.u_trans - p["u"])
            u_center = opt_lin.u.center
            opt_lin.u = opt_lin.u - u_center
            opt_lin.u_trans = VectorZonotope(
                np.vstack([f0 + u_center, np.zeros((f0.shape[0], 1), dtype=float)])
            )
            opt_lin.origin_contained = False
            # save constant input
            self._lin_error["f0"] = f0
            self._lin_error["p"] = p
            return lin_sys, opt_lin

        def _init_reach_over_lin_rem(
            self, r0, opt: NonLinearSystem.Options
        ) -> (ReachableResult, NonLinearSystem.Options):
            # compute the reachable set using the options.algo='lin' algorithm to obtain
            # a first rough over-approximation of the reachable set
            # TODO
            raise NotImplementedError

        def _linear_reach(self, opt: NonLinearSystem.Options, r: ReachableResult):
            """
            compute the reachable set after linearize the system
            :param opt: options for the computation
            :param r: given initial reachable set
            :return:

            Refs:
            [1] M. Althoff et al. "Reachability analysis of nonlinear systems with
                uncertain parameters using conservative linearization"
            [2] M. Althoff et al. "Reachability analysis of nonlinear systems using
                conservative polynomialization and non-convex sets"
            """
            # extract initial set and abstraction error
            r_init, err = r.set, r.error
            # necessary to update part of abstraction that is dependent on x0 when
            # linearization remainder is not computed
            if hasattr(opt, "update_init_func"):
                raise NotImplementedError
            # linearize the nonlinear system
            lin_sys, lin_opt = self.linearize(opt, r_init)
            # translate r_init by linearization point
            r_delta = r_init - self._lin_error["p"]["x"]
            # compute reachable set of the linearized system
            r = lin_sys.init_reach(r_delta, lin_opt)

            raise NotImplementedError

        def _init_reach_over(
            self, r_init, opt: NonLinearSystem.Options
        ) -> (ReachableResult, NonLinearSystem.Options):
            # loop over all parallel initial sets
            idx, r_tp, r_ti, r0 = 0, {}, {}, {}
            for i in range(len(r_init)):
                temp_r_ti, temp_r_tp, dim_for_split, opts = self._linear_reach(
                    opt, r_init[i]
                )
                pass

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
            # opt.validate(param)  # validate given options
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

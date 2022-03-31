from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import factorial
from sympy import lambdify

from pyrat.geometry import Geometry, VectorZonotope
from pyrat.misc import Reachable, Simulation
from pyrat.model import Model
from .continuous_system import ContSys, Option, RunTime
from .linear_system import LinSys


class NonLinSys:
    @dataclass
    class Option(Option):
        algo: str = "standard"
        taylor_terms: int = 0
        lagrange_rem = {}
        factors: np.ndarray = None
        u_trans: np.ndarray = None

        def validate(self) -> bool:
            #  TODO
            return True

    @dataclass
    class RunTime(RunTime):
        step: int = 0
        # TODO

    class Sys(ContSys):
        def __init__(self, model: Model):
            self._model = model
            self._run_time = RunTime()
            self._jx = None
            self._ju = None
            self._post_init()

        # =============================================== operator
        def __str__(self):
            raise NotImplementedError

        # =============================================== property
        def dim(self) -> int:
            raise NotImplementedError

        # =============================================== private method
        def _post_init(self):
            """
            necessary steps after general initialization
            :return:
            """
            if self._jx is None:
                self._jx = self._model.f.jacobian(self._model.vars[0])
                self._ju = self._model.f.jacobian(self._model.vars[1])

        def _evaluate(self, x, u, mod: str = "numpy"):
            f = lambdify(self._model.vars, self._model.f, mod)
            return np.squeeze(f(x, u))

        def _jacobian(self, x, u):
            fx = lambdify(self._model.vars, self._jx, "numpy")
            fu = lambdify(self._model.vars, self._ju, "numpy")
            return fx(x, u), fu(x, u)

        def _linearize(
            self, op: NonLinSys.Option, r: Geometry
        ) -> (LinSys.Sys, LinSys.Option):
            # linearization point p.u of the input is the center of the input set
            p = {"u": op.u_trans}
            # linearization point p.x of the state is the center of the last reachable
            # set R translated by 0.5*delta_t*f0
            f0_pre = self._evaluate(r.center, p["u"])
            p["x"] = r.center + f0_pre * 0.5 * op.t_step
            # substitute p into the system equation to obtain the constraint input
            f0 = self._evaluate(p["x"], p["u"])
            # substitute p into the jacobian with respect to x and u to obtain the
            # system matrix A and the input matrix B
            a, b = self._jacobian(p["x"], p["u"])
            # set up linearized system
            lin_sys = LinSys.Sys(xa=a)
            # set up options for linearized system
            # lin_op = LinSys.Option(**dataclasses.asdict(op))
            # ---------------------------------------
            lin_op = LinSys.Option()  # TODO update values inside this linear option
            lin_op.u_trans = op.u_trans
            lin_op.taylor_terms = op.taylor_terms
            lin_op.factors = op.factors
            lin_op.t_step = op.t_step
            # --------------------------------------- # TODO shall update this part
            lin_op.u = b @ (op.u + lin_op.u_trans - p["u"])
            lin_op.u -= lin_op.u.center
            lin_op.u_trans = VectorZonotope(
                np.hstack(
                    [
                        (f0 + lin_op.u.center).reshape((-1, 1)),
                        np.zeros((f0.shape[0], 1), dtype=float),
                    ]
                )
            )
            lin_op.origin_contained = False
            # save constant input
            self._run_time.lin_err_f0 = f0
            self._run_time.lin_err_p = p
            return lin_sys, lin_op

        def _lin_reach(self, r: Reachable.Element, op: NonLinSys.Option):
            # linearize the nonlinear system
            lin_sys, lin_op = self._linearize(op, r.set)
            # translate r_init by linearization point
            r_delta = r.set + (-self._run_time.lin_err_p["x"])
            # compute reachable set of linearized  system
            r = lin_sys.reach_init(r_delta, lin_op)

            exit(False)
            raise NotImplementedError

        def _reach_over_standard(self, op: NonLinSys.Option) -> Reachable.Result:
            # obtain factors for initial state and input solution time step
            i = np.arange(1, op.taylor_terms + 2)
            op.factors = np.power(op.t_step, i) / factorial(i)
            # if a trajectory should be tracked
            self._run_time.cur_t = op.t_start
            # init reachable result
            r = Reachable.Result()
            r.init(op.steps, op.steps)
            # init reachable set computation
            r_next, op = self.reach_init(op.r_init, op)
            # TODO
            raise NotImplementedError

        # TODO

        # =============================================== public method
        def reach_init(
            self, r_init: [Reachable.Element], op: NonLinSys.Option
        ) -> (Reachable.Result, NonLinSys.Option):
            # loop over all parallel initial sets
            idx, r_tp, r_ti, r0 = 0, {}, {}, {}
            for i in range(len(r_init)):
                a, b, c, d = self._lin_reach(r_init[i], op)
                raise NotImplementedError

            raise NotImplementedError

        def reach(self, op: NonLinSys.Option) -> Reachable.Result:
            assert op.validate()
            if op.algo == "lin":
                return self._reach_over_standard(op)
            else:
                raise NotImplementedError

        def simulate(self, op: NonLinSys.Option) -> Simulation.Result:
            raise NotImplementedError

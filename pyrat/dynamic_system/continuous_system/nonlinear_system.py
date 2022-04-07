from __future__ import annotations

import inspect
import numbers
from dataclasses import dataclass

import numpy as np
from scipy.special import factorial
from sympy import lambdify, hessian

from pyrat.geometry import Geometry, VectorZonotope, Interval, cvt2
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
        zonotope_order: int = 50
        reduction_method: str = "girard"
        tensor_order = 2
        max_err: np.ndarray = None

        def validate(self, dim) -> bool:
            if self.max_err is None:
                self.max_err = np.full(dim, np.inf)
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
            self._hx = None
            self._hu = None
            self._post_init()

        # =============================================== operator
        def __str__(self):
            raise NotImplementedError

        # =============================================== property
        @property
        def dim(self) -> int:
            return self._model.dim

        # =============================================== private method
        def _post_init(self):
            """
            necessary steps after general initialization
            :return:
            """
            if self._jx is None:
                self._jx = self._model.f.jacobian(self._model.vars[0])
                self._ju = self._model.f.jacobian(self._model.vars[1])
                self._hx = [
                    hessian(expr, self._model.vars[0]) for expr in self._model.f
                ]
                self._hu = [
                    hessian(expr, self._model.vars[1]) for expr in self._model.f
                ]  # TODO need refine this part, check "sympy array derivative"

        def _evaluate(self, x, u, mod: str = "numpy"):
            f = lambdify(self._model.vars, self._model.f, mod)
            return np.squeeze(f(x, u))

        def _jacobian(self, x, u):
            fx = lambdify(self._model.vars, self._jx, "numpy")
            fu = lambdify(self._model.vars, self._ju, "numpy")
            return fx(x, u), fu(x, u)

        def _hessian(self, x, u):
            ops = Interval.ops()

            def _fill_hessian(expr_h, dim):
                hs = []
                for expr in expr_h:
                    h = np.zeros((2, dim, dim), dtype=float)
                    for idx in range(len(expr)):
                        row, col = int(idx / x.dim[0]), int(idx % x.dim[0])
                        if not expr[idx].is_number:
                            f = lambdify(self._model.vars, expr[idx], ops)
                            v = f(x, u)
                            h[0, row, col] = v.inf
                            h[1, row, col] = v.sup
                    hs.append(Interval(h))
                return hs

            hx = _fill_hessian(self._hx, x.dim[0])
            hu = _fill_hessian(self._hu, u.dim[0])

            return hx, hu

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
            lin_op.zonotope_order = op.zonotope_order
            # --------------------------------------- # TODO shall refine this part, like unify the option???
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

        def _abst_err_lin(
            self, op: NonLinSys.Option, r: Geometry
        ) -> (Interval, VectorZonotope):
            """
            computes the abstraction error for linearization approach to enter
            :param op:
            :param r:
            :return:
            """
            # compute interval of reachable set
            ihx = cvt2(r, "int")
            # compute intervals of total reachable set
            total_int_x = ihx + self._run_time.lin_err_p["x"]

            # compute intervals of input
            ihu = cvt2(op.u, "int")
            # translate intervals by linearization point
            total_int_u = ihu + self._run_time.lin_err_p["u"]

            if op.tensor_order == 2:
                # obtain maximum absolute values within ihx, ihu
                dx = np.maximum(abs(ihx.inf), abs(ihx.sup))
                du = np.maximum(abs(ihu.inf), abs(ihu.sup))

                # evaluate the hessian matrix with the selected range-bounding technique
                hx, hu = self._hessian(total_int_x, total_int_u)

                # calculate the Lagrange remainder (second-order error)
                err_lagr = np.zeros(self.dim, dtype=float)

                for i in range(self.dim):
                    abs_hx, abs_hu = abs(hx[i]), abs(hu[i])
                    hx_ = np.max(abs_hx.bd, axis=0)
                    hu_ = np.max(abs_hu.bd, axis=0)
                    err_lagr[i] = 0.5 * (dx @ hx_ @ dx + du @ hu_ @ du)

                v_err_dyn = VectorZonotope(
                    np.hstack([0 * err_lagr.reshape((-1, 1)), np.diag(err_lagr)])
                )
                return err_lagr, v_err_dyn
            else:
                raise NotImplementedError  # TODO

        def _lin_reach(self, r_init: Reachable.Element, op: NonLinSys.Option):
            # linearize the nonlinear system
            lin_sys, lin_op = self._linearize(op, r_init.set)
            # translate r_init by linearization point
            r_delta = r_init.set + (-self._run_time.lin_err_p["x"])
            # compute reachable set of linearized  system
            r, _ = lin_sys.reach_init(r_delta, lin_op)

            # compute reachable set of the abstracted system including the abstraction
            # error using the selected algorithm

            if op.algo == "lin_rem":
                raise NotImplementedError
            else:
                # loop until the actual abstraction error is smaller than the estimated
                # linearization error
                r_tp, r_ti = r.tp, r.ti
                perf_ind_cur, perf_ind = np.inf, 0
                applied_err, abstr_err, v_err_dyn = None, r_init.err, None

                while perf_ind_cur > 1 and perf_ind <= 1:
                    # estimate the abstraction error
                    applied_err = 1.1 * abstr_err
                    v_err = VectorZonotope(
                        np.hstack(
                            [0 * applied_err.reshape((-1, 1)), np.diag(applied_err)]
                        )
                    )
                    r_all_err = lin_sys.error_solution(lin_op, v_err)

                    # compute the abstraction error using the conservative linearization
                    # approach described in [1]
                    if op.algo == "lin":
                        # compute overall reachable set including linearization error
                        r_max = r_ti + r_all_err
                        # compute linearization error
                        true_err, v_err_dyn = self._abst_err_lin(op, r_max)
                    else:
                        raise NotImplementedError  # TODO

                    # compare linearization error with the maximum allowed error
                    perf_ind_cur = np.max(true_err / applied_err)
                    perf_ind = np.max(true_err / op.max_err)
                    abstr_err = true_err

                # translate reachable sets by linearization point
                r_ti += self._run_time.lin_err_p["x"]
                r_tp += self._run_time.lin_err_p["x"]

                # compute the reachable set due to the linearization error
                r_err = lin_sys.error_solution(lin_op, v_err_dyn)

                # add the abstraction error to the reachable sets
                r_ti += r_err
                r_tp += r_err
            # determine the best dimension to split the set in order to reduce the
            # linearization error
            dim_for_split = []
            if perf_ind > 1:
                raise NotImplementedError  # TODO
            # store the linearization error
            return r_ti, Reachable.Element(r_tp, abstr_err), dim_for_split

        def _reach_over_standard(self, op: NonLinSys.Option) -> Reachable.Result:
            # obtain factors for initial state and input solution time step
            i = np.arange(1, op.taylor_terms + 2)
            op.factors = np.power(op.t_step, i) / factorial(i)
            # if a trajectory should be tracked
            self._run_time.cur_t = op.t_start
            # init containers for storing the results
            ti_set, ti_time, tp_set, tp_time = {}, {}, {}, {}

            # init reachable set computation
            r_next = self.reach_init(op.r_init, op)

            # loop over all reachability steps
            for i in range(op.steps):
                # save reachable set
                ti_set[i] = r_next[0]
                # ti_time[i]=Interval(np.array([[]]))
                # TODO
                raise NotImplementedError

            # TODO
            raise NotImplementedError

            # =============================================== public method

        def reach_init(
            self, r_init: [Reachable.Element], op: NonLinSys.Option
        ) -> (Reachable.Result, NonLinSys.Option):
            # loop over all parallel initial sets
            idx, r_tp, r_ti, r0 = 0, {}, {}, {}
            for i in range(len(r_init)):
                temp_r_ti, temp_r_tp, dim_for_split = self._lin_reach(r_init[i], op)

                # check if initial set has to be split
                if len(dim_for_split) <= 0:
                    r_tp[idx] = temp_r_tp
                    r_tp[idx].pre = i
                    r_ti[idx] = temp_r_ti
                    r0[idx] = r_init[i]
                    idx += 1
                else:
                    raise NotImplementedError  # TODO

            # store the result
            return r_ti, r_tp, r0

        def reach(self, op: NonLinSys.Option) -> Reachable.Result:
            assert op.validate(self.dim)
            if op.algo == "lin":
                return self._reach_over_standard(op)
            else:
                raise NotImplementedError

        def simulate(self, op: NonLinSys.Option) -> Simulation.Result:
            raise NotImplementedError

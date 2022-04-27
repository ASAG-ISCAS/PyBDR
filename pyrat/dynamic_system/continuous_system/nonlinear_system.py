from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy
from scipy.special import factorial
from sympy import lambdify, hessian, derive_by_array

from pyrat.geometry import Geometry, Zonotope, Interval, IntervalMatrix, cvt2
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
        reduction_method: str = Zonotope.MethodReduce.GIRARD
        tensor_order = 2
        max_err: np.ndarray = None

        def validate(self, dim) -> bool:
            assert self.steps >= 1
            if self.max_err is None:
                self.max_err = np.full(dim, np.inf)
            if self.step_size is None:
                self.step_size = (self.t_end - self.t_start) / (self.steps - 1)

            #  TODO
            return True

    @dataclass
    class RunTime(RunTime):
        step: int = 0
        # TODO

    class Sys(ContSys):
        def __init__(self, model: Model):
            assert 1 <= len(model.vars) <= 2  # only support f(x) or f(x,u) as input
            self._model = model
            self._run_time = RunTime()
            self._jaco = None
            self._hess = None
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
            def _take_derivative(f, x):
                d = np.asarray(derive_by_array(f, x))
                return np.moveaxis(d, 0, -2)

            self._jaco, self._hess = [], []
            for var in self._model.vars:
                j = _take_derivative(self._model.f, var)
                h = _take_derivative(j, var)
                j = j.squeeze(axis=-1)
                h = h.squeeze(axis=-1)
                self._jaco.append(sympy.ImmutableDenseNDimArray(j))
                self._hess.append(sympy.ImmutableDenseNDimArray(h))

        def _evaluate(self, xs: tuple, mod: str = "numpy"):
            f = lambdify(self._model.vars, self._model.f, mod)
            return np.squeeze(f(*xs))

        def _jacobian(self, xs: tuple, mod: str = "numpy"):
            def _eval_jacob(jc):
                f = lambdify(self._model.vars, jc, mod)
                return f(*xs)

            return [_eval_jacob(j) for j in self._jaco]

        def _hessian(self, xs: tuple, ops: dict):
            """
            NOTE: only support interval matrix currently
            """

            def _eval_hessian_interval(he):
                def __eval_element(x):
                    if x.is_number:
                        return Interval(float(x), float(x))
                    else:
                        f = lambdify(self._model.vars, x, ops)
                        return f(*xs)

                v = np.vectorize(__eval_element)(he)
                inf = np.vectorize(lambda x: x.inf)(v)
                sup = np.vectorize(lambda x: x.sup)(v)
                return [
                    IntervalMatrix(inf[idx], sup[idx]) for idx in range(he.shape[0])
                ]

            return [_eval_hessian_interval(h) for h in self._hess]

        # ------------------------------------------------------------------------------

        def _linearize(
            self, op: NonLinSys.Option, r: Geometry.Base
        ) -> (LinSys.Sys, LinSys.Option):
            # linearization point p.u of the input is the center of the input set
            p = {"u": op.u_trans} if op.u is not None else {}
            # linearization point p.x of the state is the center of the last reachable
            # set R translated by 0.5*delta_t*f0
            f0_pre = self._evaluate((r.c, p["u"]))
            p["x"] = r.c + f0_pre * 0.5 * op.step_size
            # substitute p into the system equation to obtain the constraint input
            f0 = self._evaluate((p["x"], p["u"]))
            # substitute p into the jacobian with respect to x and u to obtain the
            # system matrix A and the input matrix B
            a, b = self._jacobian((p["x"], p["u"]))
            assert not (np.any(np.isnan(a)) or np.any(np.isnan(b)))
            # set up linearized system
            lin_sys = LinSys.Sys(xa=a)
            # set up options for linearized system
            # ---------------------------------------
            lin_op = LinSys.Option()  # TODO update values inside this linear option
            lin_op.u_trans = op.u_trans
            lin_op.taylor_terms = op.taylor_terms
            lin_op.factors = op.factors
            lin_op.step_size = op.step_size
            lin_op.zonotope_order = op.zonotope_order
            lin_op.cur_t = op.cur_t
            # --------------------------------------- # TODO shall refine this part, like unify the option???
            lin_op.u = b @ (op.u + lin_op.u_trans - p["u"])
            lin_op.u -= lin_op.u.c
            lin_op.u_trans = Zonotope(
                f0 + lin_op.u.c, np.zeros((f0.shape[0], 1), dtype=float)
            )
            lin_op.origin_contained = False
            # save constant input
            self._run_time.lin_err_f0 = f0
            self._run_time.lin_err_p = p
            return lin_sys, lin_op

        def _abst_err_lin(
            self, op: NonLinSys.Option, r: Geometry
        ) -> (Interval, Zonotope):
            """
            computes the abstraction error for linearization approach to enter
            :param op:
            :param r:
            :return:
            """
            # compute interval of reachable set
            ihx = cvt2(r, Geometry.TYPE.INTERVAL)
            # compute intervals of total reachable set
            total_int_x = ihx + self._run_time.lin_err_p["x"]

            # compute intervals of input
            ihu = cvt2(op.u, Geometry.TYPE.INTERVAL)
            # translate intervals by linearization point
            total_int_u = ihu + self._run_time.lin_err_p["u"]

            if op.tensor_order == 2:
                # obtain maximum absolute values within ihx, ihu
                dx = np.maximum(abs(ihx.inf), abs(ihx.sup))
                du = np.maximum(abs(ihu.inf), abs(ihu.sup))

                # evaluate the hessian matrix with the selected range-bounding technique
                hx, hu = self._hessian(
                    (total_int_x, total_int_u), Interval.functional()
                )

                # calculate the Lagrange remainder (second-order error)
                err_lagr = np.zeros(self.dim, dtype=float)

                for i in range(self.dim):
                    abs_hx, abs_hu = abs(hx[i]), abs(hu[i])
                    hx_ = np.maximum(abs_hx.inf, abs_hx.sup)
                    hu_ = np.maximum(abs_hu.inf, abs_hu.sup)
                    err_lagr[i] = 0.5 * (dx @ hx_ @ dx + du @ hu_ @ du)

                v_err_dyn = Zonotope(0 * err_lagr.reshape(-1), np.diag(err_lagr))
                return err_lagr, v_err_dyn
            elif op.tensor_order == 3:
                raise NotImplementedError  # TODO
            else:
                raise Exception("unsupported tensor order")

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
                    v_err = Zonotope(0 * applied_err, np.diag(applied_err))
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

        def _post(self, r: [Reachable.Element], op: NonLinSys.Option):
            """
            computes the reachable continuous set for one time step of a nonlinear system
            by over approximate linearization
            :param r:
            :param op:
            :return:
            """

            next_ti, next_tp, next_r0 = self._reach_init(r, op)

            # reduce zonotopes
            for i in range(len(next_tp)):
                if not next_tp[i].set.is_empty:
                    next_tp[i].set.reduce(op.reduction_method, op.zonotope_order)
                    next_ti[i].reduce(op.reduction_method, op.zonotope_order)

            # delete redundant reachable sets
            # TODO

            return next_ti, next_tp, next_r0

        def _reach_init(
            self, r_init: [Reachable.Element], op: NonLinSys.Option
        ) -> ([Geometry], [Reachable.Element], [Reachable.Element]):
            # loop over all parallel initial sets
            r_ti, r_tp, r0 = [], [], []
            for i in range(len(r_init)):
                temp_r_ti, temp_r_tp, dim_for_split = self._lin_reach(r_init[i], op)

                # check if initial set has to be split
                if len(dim_for_split) <= 0:
                    r_tp.append(temp_r_tp)
                    r_tp[-1].pre = i
                    r_ti.append(temp_r_ti)
                    r0.append(r_init[i])
                else:
                    raise NotImplementedError  # TODO

            # store the result
            return r_ti, r_tp, r0

        def _reach_over_standard(self, op: NonLinSys.Option) -> Reachable.Result:
            # obtain factors for initial state and input solution time step
            i = np.arange(1, op.taylor_terms + 2)
            op.factors = np.power(op.step_size, i) / factorial(i)
            # set current time
            op.cur_t = op.t_start
            # init containers for storing the results
            ti_set, ti_time, tp_set, tp_time = [], [], [], []

            # init reachable set computation
            next_ti, next_tp, next_r0 = self._reach_init(op.r_init, op)

            time_pts = np.linspace(op.t_start, op.t_end, op.steps, endpoint=True)

            # loop over all reachability steps
            for i in range(op.steps - 1):
                # save reachable set
                ti_set.append(next_ti)
                ti_time.append(time_pts[i : i + 2])
                tp_set.append(next_tp)
                tp_time.append(time_pts[i + 1])

                # check specification
                if op.specs is not None:
                    raise NotImplementedError  # TODO

                # increment time
                op.cur_t = time_pts[i + 1]

                # compute next reachable set
                next_ti, next_tp, next_r0 = self._post(next_tp, op)

            # check specification
            if op.specs is not None:
                raise NotImplementedError

            # save the last reachable set in cell structure
            return Reachable.Result(
                ti_set, tp_set, np.vstack(ti_time), np.array(tp_time)
            )

        def _reach_over_poly(self, op: NonLinSys.Option) -> Reachable.Result:
            raise NotImplementedError  # TODO

            # =============================================== public method

        def reach(self, op: NonLinSys.Option) -> Reachable.Result:
            assert op.validate(self.dim)
            if op.algo == "lin":
                return self._reach_over_standard(op)
            elif op.algo == "poly":
                return self._reach_over_poly(op)
            else:
                raise NotImplementedError

        def simulate(self, op: NonLinSys.Option) -> Simulation.Result:
            raise NotImplementedError

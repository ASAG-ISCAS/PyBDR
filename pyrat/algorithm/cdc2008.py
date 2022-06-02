"""
REF:

Althoff, M., Stursberg, O., & Buss, M. (2008, December). Reachability analysis of
nonlinear systems with uncertain parameters using conservative linearization. In 2008
47th IEEE Conference on Decision and Control (pp. 4042-4048). IEEE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import factorial

from pyrat.dynamic_system import LinSys, NonLinSys
from pyrat.geometry import Geometry, Zonotope
from pyrat.geometry.operation import cvt2
from pyrat.misc import Set, Reachable
from .algorithm import Algorithm
from .hscc2011 import HSCC2011


class CDC2008:
    @dataclass
    class Options(Algorithm.Options):
        taylor_terms: int = 4
        tensor_order: int = 2
        u_trans: np.ndarray = None
        factors: np.ndarray = None
        max_err: np.ndarray = None
        lin_err_x = None
        lin_err_u = None
        lin_err_f0 = None

        def _validate_misc(self, dim: int):
            assert self.tensor_order == 2 or self.tensor_order == 3
            self.max_err = (
                np.full(dim, np.inf) if self.max_err is None else self.max_err
            )
            i = np.arange(1, self.taylor_terms + 2)
            self.factors = np.power(self.step, i) / factorial(i)
            return True

        def validation(self, dim: int):
            assert self._validate_time_related()
            assert self._validate_inputs()
            assert self._validate_misc(dim)
            return True

    @staticmethod
    def linearize(sys: NonLinSys.Entity, r: Geometry.Base, opt: Options):
        opt.lin_err_u = opt.u_trans if opt.u_trans is not None else opt.u.c
        f0 = sys.evaluate((r.c, opt.lin_err_u))
        opt.lin_err_x = r.c + f0 * 0.5 * opt.step
        opt.lin_err_f0 = sys.evaluate((opt.lin_err_x, opt.lin_err_u))
        a, b = sys.jacobian((opt.lin_err_x, opt.lin_err_u))
        assert not (np.any(np.isnan(a))) or np.any(np.isnan(b))
        lin_sys = LinSys.Entity(xa=a)
        lin_opt = HSCC2011.Options()
        lin_opt.step = opt.step
        lin_opt.taylor_terms = opt.taylor_terms
        lin_opt.factors = opt.factors
        lin_opt.u = b @ (opt.u + opt.u_trans - opt.lin_err_u)
        lin_opt.u -= lin_opt.u.c
        lin_opt.u_trans = Zonotope(
            opt.lin_err_f0 + lin_opt.u.c, np.zeros((opt.lin_err_f0.shape[0], 1))
        )
        return lin_sys, lin_opt

    @staticmethod
    def abstract_err(sys: NonLinSys.Entity, r: Geometry.Base, opt: Options):
        ihx = cvt2(r, Geometry.TYPE.INTERVAL)
        total_int_x = ihx + opt.lin_err_x

        ihu = cvt2(opt.u, Geometry.TYPE.INTERVAL)
        total_int_u = ihu + opt.lin_err_u

        if opt.tensor_order == 2:
            dx = np.maximum(abs(ihx.inf), abs(ihx.sup))
            du = np.maximum(abs(ihu.inf), abs(ihu.sup))

            # evaluate the hessian matrix with the selected range-bounding technique
            hx, hu = sys.hessian((total_int_x, total_int_u), "interval")

            err_lagrange = np.zeros(sys.dim, dtype=float)

            for i in range(sys.dim):
                abs_hx, abs_hu = abs(hx[i]), abs(hu[i])
                hx_ = np.maximum(abs_hx.inf, abs_hx.sup)
                hu_ = np.maximum(abs_hu.inf, abs_hu.sup)
                err_lagrange[i] = 0.5 * (dx @ hx_ @ dx + du @ hu_ @ du)
            v_err_dyn = Zonotope(np.zeros(sys.dim), np.diag(err_lagrange))
            return err_lagrange, v_err_dyn
        elif opt.tensor_order == 3:
            raise NotImplementedError  # TODO
        else:
            raise Exception("unsupported tensor order")

    @classmethod
    def linear_reach(cls, sys: NonLinSys.Entity, r: Set, opt: Options):
        lin_sys, lin_opt = cls.linearize(sys, r.geometry, opt)
        r_delta = r.geometry - opt.lin_err_x
        r_ti, r_tp = HSCC2011.reach_one_step(lin_sys, r_delta, lin_opt)

        perf_ind_cur, perf_ind = np.inf, 0
        applied_err, abstract_err, v_err_dyn = None, r.err, None

        while perf_ind_cur > 1 and perf_ind <= 1:
            applied_err = 1.1 * abstract_err
            v_err = Zonotope(0 * applied_err, np.diag(applied_err))
            r_all_err = HSCC2011.error_solution(v_err, lin_opt)
            r_max = r_ti + r_all_err
            true_err, v_err_dyn = cls.abstract_err(sys, r_max, opt)

            # compare linearization error with the maximum allowed error
            temp = true_err / applied_err
            temp[np.isnan(temp)] = -np.inf
            perf_ind_cur = np.max(temp)
            perf_ind = np.max(true_err / opt.max_err)
            abstract_err = true_err

            # exception for set explosion
            if np.any(abstract_err > 1e100):
                raise Exception("Set Explosion")
        # translate reachable sets by linearization point
        r_ti += opt.lin_err_x
        r_tp += opt.lin_err_x

        # compute the reachable set due to the linearization error
        r_err = HSCC2011.error_solution(v_err_dyn, lin_opt)

        # add the abstraction error to the reachable sets
        r_ti += r_err
        r_tp += r_err
        # determine the best dimension to split the set in order to reduce the
        # linearization error
        dim_for_split = []
        if perf_ind > 1:
            raise NotImplementedError  # TODO
        # store the linearization error
        r_ti = r_ti.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
        r_tp = r_tp.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
        return r_ti, Set(r_tp, abstract_err), dim_for_split

    @classmethod
    def reach_one_step(cls, sys: NonLinSys.Entity, r0: [Set], opt: Options):
        r_ti, r_tp = [], []
        for i in range(len(r0)):
            temp_r_ti, temp_r_tp, dims = cls.linear_reach(sys, r0[i], opt)
            # check if initial set has to be split
            if len(dims) <= 0:
                r_ti.append(temp_r_ti)
                r_tp.append(temp_r_tp)
            else:
                raise NotImplementedError  # TODO

        return r_ti, r_tp

    @classmethod
    def reach(cls, sys: NonLinSys.Entity, opt: Options):
        assert opt.validation(sys.dim)
        # init containers for storing the results
        time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
        ti_set, ti_time, tp_set, tp_time = [], [], [opt.r0], [time_pts[0]]

        while opt.step_idx < opt.steps_num - 1:
            next_ti, next_tp = cls.reach_one_step(sys, tp_set[-1], opt)
            opt.step_idx += 1
            ti_set.append(next_ti)
            ti_time.append(time_pts[opt.step_idx - 1 : opt.step_idx + 1])
            tp_set.append(next_tp)
            tp_time.append(time_pts[opt.step_idx])

        return Reachable.Result(ti_set, tp_set, np.vstack(ti_time), np.array(tp_time))

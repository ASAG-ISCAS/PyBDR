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

from pybdr.dynamic_system import LinSys, NonLinSys
from pybdr.geometry import Geometry, Zonotope, Interval
from pybdr.geometry.operation import cvt2
from .algorithm import Algorithm
from .alk2011hscc import ALK2011HSCC


class ASB2008CDC:
    @dataclass
    class Options(Algorithm.Options):
        taylor_terms: int = 4  # for linearization
        tensor_order: int = 2  # for error approximation
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
            assert self._validate_misc(dim)
            return True

    @staticmethod
    def linearize(sys: NonLinSys, r: Geometry.Base, opt: Options):
        opt.lin_err_u = opt.u_trans if opt.u_trans is not None else opt.u.c
        f0 = sys.evaluate((r.c, opt.lin_err_u), "numpy", 0, 0)
        opt.lin_err_x = r.c + f0 * 0.5 * opt.step
        opt.lin_err_f0 = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 0, 0)
        a = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 1, 0)
        b = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 1, 1)
        assert not (np.any(np.isnan(a))) or np.any(np.isnan(b))
        lin_sys = LinSys(xa=a)
        lin_opt = ALK2011HSCC.Options()
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
    def abstract_err(sys: NonLinSys, r: Geometry.Base, opt: Options):
        ihx = cvt2(r, Geometry.TYPE.INTERVAL)
        total_int_x = ihx + opt.lin_err_x

        ihu = cvt2(opt.u, Geometry.TYPE.INTERVAL)
        total_int_u = ihu + opt.lin_err_u

        if opt.tensor_order == 2:
            dx = np.maximum(abs(ihx.inf), abs(ihx.sup))
            du = np.maximum(abs(ihu.inf), abs(ihu.sup))

            # evaluate the hessian matrix with the selected range-bounding technique
            hx = sys.evaluate((total_int_x, total_int_u), "interval", 2, 0)
            hu = sys.evaluate((total_int_x, total_int_u), "interval", 2, 1)
            xx = np.maximum(abs(hx.inf), abs(hx.sup))
            uu = np.maximum(abs(hu.inf), abs(hu.sup))

            err_lagrange = 0.5 * (dx @ xx @ dx + du @ uu @ du)

            verr_dyn = Zonotope(np.zeros(sys.dim), np.diag(err_lagrange))
            return err_lagrange, verr_dyn
        elif opt.tensor_order == 3:
            r_red = r.reduce(Zonotope.REDUCE_METHOD, Zonotope.ERROR_ORDER)
            z = r_red.card_prod(opt.u)
            # evaluate hessian
            hx = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 2, 0)
            hu = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 2, 1)
            # evaluate third order
            tx = sys.evaluate((total_int_x, total_int_u), "interval", 3, 0)
            tu = sys.evaluate((total_int_x, total_int_u), "interval", 3, 1)

            # second order error
            err_sec = 0.5 * z.quad_map([hx, hu])
            xx = Interval.sum((ihx @ tx @ ihx) * ihx, axis=1)
            uu = Interval.sum((ihu @ tu @ ihu) * ihu, axis=1)
            err_lagr = (xx + uu) / 6
            err_lagr = cvt2(err_lagr, Geometry.TYPE.ZONOTOPE)

            # overall linearization error
            verr_dyn = err_sec + err_lagr
            verr_dyn = verr_dyn.reduce(
                Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER
            )
            true_err = abs(cvt2(verr_dyn, Geometry.TYPE.INTERVAL)).sup
            return true_err, verr_dyn
        else:
            raise Exception("unsupported tensor order")

    @classmethod
    def linear_reach(cls, sys: NonLinSys, r, err, opt: Options):
        lin_sys, lin_opt = cls.linearize(sys, r, opt)
        r_delta = r - opt.lin_err_x
        r_ti, r_tp = ALK2011HSCC.reach_one_step(lin_sys, r_delta, lin_opt)

        perf_ind_cur, perf_ind = np.inf, 0
        applied_err, abstract_err, v_err_dyn = None, err, None

        while perf_ind_cur > 1 and perf_ind <= 1:
            applied_err = 1.1 * abstract_err
            v_err = Zonotope(0 * applied_err, np.diag(applied_err))
            r_all_err = ALK2011HSCC.error_solution(v_err, lin_opt)
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
        r_err = ALK2011HSCC.error_solution(v_err_dyn, lin_opt)

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
        return r_ti, r_tp, abstract_err, dim_for_split

    @classmethod
    def reach_one_step(cls, sys: NonLinSys, r0, err, opt: Options):
        r_ti, r_tp = [], []
        for i in range(len(r0)):
            temp_r_ti, temp_r_tp, abst_err, dims = cls.linear_reach(
                sys, r0[i], err[i], opt
            )
            # check if initial set has to be split
            if len(dims) <= 0:
                r_ti.append(temp_r_ti)
                r_tp.append(temp_r_tp)
            else:
                raise NotImplementedError  # TODO

        return r_ti, r_tp

    @classmethod
    def reach(cls, sys: NonLinSys, opt: Options):
        assert opt.validation(sys.dim)
        # init containers for storing the results
        time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
        ti_set, ti_time, tp_set, tp_time = [], [], [opt.r0], [time_pts[0]]
        err = [[np.zeros(r.shape) for r in opt.r0]]

        while opt.step_idx < opt.steps_num - 1:
            next_ti, next_tp = cls.reach_one_step(sys, tp_set[-1], err[-1], opt)
            opt.step_idx += 1
            ti_set.append(next_ti)
            ti_time.append(time_pts[opt.step_idx - 1 : opt.step_idx + 1])
            tp_set.append(next_tp)
            tp_time.append(time_pts[opt.step_idx])

        return ti_set, tp_set, np.vstack(ti_time), np.array(tp_time)

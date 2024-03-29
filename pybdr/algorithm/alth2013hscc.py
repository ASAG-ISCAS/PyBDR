"""
REF:

Althoff, M. (2013, April). Reachability analysis of nonlinear systems using conservative
 polynomialization and non-convex sets. In Proceedings of the 16th international
 conference on Hybrid systems: computation and control (pp. 173-182).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable
from functools import partial

from scipy.special import factorial
from pybdr.model import Model
from pybdr.dynamic_system import NonLinSys
from pybdr.geometry import Geometry, Zonotope, Interval
from pybdr.geometry.operation import cvt2
from .algorithm import Algorithm
from .alk2011hscc import ALK2011HSCC
from .asb2008cdc import ASB2008CDC


class ALTH2013HSCC:
    class Options(Algorithm.Options):
        taylor_terms: int = 4
        tensor_order: int = 3
        u_trans: np.ndarray = None
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
            assert 3 <= self.tensor_order <= 7
            return True

        def validation(self, dim: int):
            assert self._validate_time_related()
            # assert self._validate_inputs()
            assert self._validate_misc(dim)
            return True

    @classmethod
    def pre_stat_err(cls, dyn: Callable, dims, r_delta: Zonotope, opt: Options):
        sys = Model(dyn, dims)
        r_red = cvt2(r_delta, Geometry.TYPE.ZONOTOPE).reduce(Zonotope.REDUCE_METHOD, Zonotope.ERROR_ORDER)
        # extend the sets by the input sets
        u_stat = Zonotope.zero(opt.u.shape)
        z = r_red.card_prod(u_stat)
        z_delta = r_delta.card_prod(u_stat)
        # compute hessian
        hx = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 2, 0)
        hu = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 2, 1)

        t, ind3, zd3 = None, None, None

        # calculate the quadratic map == static second order error
        err_stat_sec = 0.5 * z.quad_map([hx, hu])
        err_stat = None
        # third order tensor
        if opt.tensor_order >= 4:
            raise NotImplementedError
        else:
            err_stat = err_stat_sec
            err_stat = err_stat.reduce(
                Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER
            )
        return [hx, hu], z_delta, err_stat, t, ind3, zd3

    # @classmethod
    # def pre_stat_err(cls, sys: NonLinSys, r_delta: Zonotope, opt: Options):
    #     r_red = cvt2(r_delta, Geometry.TYPE.ZONOTOPE).reduce(
    #         Zonotope.REDUCE_METHOD, Zonotope.ERROR_ORDER
    #     )
    #     # extend teh sets byt the input sets
    #     u_stat = Zonotope.zero(opt.u.shape)
    #     z = r_red.card_prod(u_stat)
    #     z_delta = r_delta.card_prod(u_stat)
    #     # compute hessian
    #     hx = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 2, 0)
    #     hu = sys.evaluate((opt.lin_err_x, opt.lin_err_u), "numpy", 2, 1)
    #
    #     t, ind3, zd3 = None, None, None
    #
    #     # calculate the quadratic map == static second order error
    #     err_stat_sec = 0.5 * z.quad_map([hx, hu])
    #     err_stat = None
    #     # third order tensor
    #     if opt.tensor_order >= 4:
    #         raise NotImplementedError
    #     else:
    #         err_stat = err_stat_sec
    #         err_stat = err_stat.reduce(
    #             Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER
    #         )
    #     return [hx, hu], z_delta, err_stat, t, ind3, zd3

    @classmethod
    def abst_err(cls, dyn: Callable, dims, opt, r_all, r_diff, h, zd, verr_stat):
        sys = Model(dyn, dims)
        # compute interval of the reachable set
        dx = cvt2(r_all, Geometry.TYPE.INTERVAL)
        total_int_x = dx + opt.lin_err_x
        # compute intervals of input
        du = cvt2(opt.u, Geometry.TYPE.INTERVAL)
        total_int_u = du + opt.lin_err_u

        # compute zonotope of state and input
        r_red_diff = cvt2(r_diff, Geometry.TYPE.ZONOTOPE).reduce(
            Zonotope.REDUCE_METHOD, Zonotope.ERROR_ORDER
        )
        z_diff = r_red_diff.card_prod(opt.u)

        # second order error
        err_dyn_sec = 0.5 * (
                zd.quad_map(h, z_diff) + z_diff.quad_map(h, zd) + z_diff.quad_map(h)
        )

        if opt.tensor_order == 3:
            tx = sys.evaluate((total_int_x, total_int_u), "interval", 3, 0)
            tu = sys.evaluate((total_int_x, total_int_u), "interval", 3, 1)

            xx = Interval.sum((dx @ tx @ dx) * dx, axis=1)
            uu = Interval.sum((du @ tu @ du) * du, axis=1)
            err_dyn_third = (xx + uu) / 6
            err_dyn_third = cvt2(err_dyn_third, Geometry.TYPE.ZONOTOPE)

            # no terms of order >=4, max 3 for now
            remainder = Zonotope.zero(sys.dim, 1)

        else:
            raise NotImplementedError
        verr_dyn = err_dyn_sec + err_dyn_third + remainder
        verr_dyn = verr_dyn.reduce(Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER)

        err_ih_abs = abs(
            cvt2(verr_dyn, Geometry.TYPE.INTERVAL)
            + cvt2(verr_stat, Geometry.TYPE.INTERVAL)
        )
        true_err = err_ih_abs.sup
        return true_err, verr_dyn, verr_stat

    # @classmethod
    # def abstract_err(cls, sys, opt, r_all, r_diff, h, zd, verr_stat, t, ind3, zd3):
    #     # compute interval of reachable set
    #     dx = cvt2(r_all, Geometry.TYPE.INTERVAL)
    #     total_int_x = dx + opt.lin_err_x
    #     # compute intervals of input
    #     du = cvt2(opt.u, Geometry.TYPE.INTERVAL)
    #     total_int_u = du + opt.lin_err_u
    #
    #     # compute zonotope of state and input
    #     r_red_diff = cvt2(r_diff, Geometry.TYPE.ZONOTOPE).reduce(
    #         Zonotope.REDUCE_METHOD, Zonotope.ERROR_ORDER
    #     )
    #     z_diff = r_red_diff.card_prod(opt.u)
    #
    #     # second order error
    #     err_dyn_sec = 0.5 * (
    #             zd.quad_map(h, z_diff) + z_diff.quad_map(h, zd) + z_diff.quad_map(h)
    #     )
    #
    #     if opt.tensor_order == 3:
    #         tx = sys.evaluate((total_int_x, total_int_u), "interval", 3, 0)
    #         tu = sys.evaluate((total_int_x, total_int_u), "interval", 3, 1)
    #
    #         xx = Interval.sum((dx @ tx @ dx) * dx, axis=1)
    #         uu = Interval.sum((du @ tu @ du) * du, axis=1)
    #         err_dyn_third = (xx + uu) / 6
    #         err_dyn_third = cvt2(err_dyn_third, Geometry.TYPE.ZONOTOPE)
    #
    #         # no terms of order >=4, max 3 for now
    #         remainder = Zonotope.zero(sys.dim, 1)
    #
    #     else:
    #         raise NotImplementedError
    #     verr_dyn = err_dyn_sec + err_dyn_third + remainder
    #     verr_dyn = verr_dyn.reduce(Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER)
    #
    #     err_ih_abs = abs(
    #         cvt2(verr_dyn, Geometry.TYPE.INTERVAL)
    #         + cvt2(verr_stat, Geometry.TYPE.INTERVAL)
    #     )
    #     true_err = err_ih_abs.sup
    #     return true_err, verr_dyn, verr_stat

    @classmethod
    def poly_reach(cls, dyn: Callable, dims, r, err, opt: Options):
        lin_sys, lin_opt = ASB2008CDC.linearize(dyn, dims, r, opt)
        r_delta = r - opt.lin_err_x
        r_ti, r_tp = ALK2011HSCC.reach_one_step(lin_sys, r_delta, lin_opt)
        r_diff = ALK2011HSCC.delta_reach(lin_sys, r_delta, lin_opt)
        h, zd, err_stat, t, ind3, zd3 = cls.pre_stat_err(dyn, dims, r_delta, opt)
        perf_ind_cur, perf_ind = np.inf, 0
        applied_err, abstract_err, v_err_dyn, v_err_stat = None, err, None, None

        while perf_ind_cur > 1 and perf_ind <= 1:
            applied_err = 1.1 * abstract_err
            v_err = Zonotope(0 * applied_err, np.diag(applied_err))
            r_all_err = ALK2011HSCC.error_solution(v_err, lin_opt)
            r_max = r_delta + cvt2(r_diff, Geometry.TYPE.ZONOTOPE) + r_all_err
            true_err, v_err_dyn, v_err_stat = cls.abst_err(dyn, dims, opt, r_max, r_diff + r_all_err, h, zd, err_stat)

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

    # @classmethod
    # def poly_reach(cls, sys: NonLinSys, r, err, opt: Options):
    #     lin_sys, lin_opt = ASB2008CDC.linearize(sys, r, opt)
    #     r_delta = r - opt.lin_err_x
    #     r_ti, r_tp = ALK2011HSCC.reach_one_step(lin_sys, r_delta, lin_opt)
    #     r_diff = ALK2011HSCC.delta_reach(lin_sys, r_delta, lin_opt)
    #     h, zd, err_stat, t, ind3, zd3 = cls.pre_stat_err(sys, r_delta, opt)
    #     perf_ind_cur, perf_ind = np.inf, 0
    #     applied_err, abstract_err, v_err_dyn, v_err_stat = None, err, None, None
    #
    #     while perf_ind_cur > 1 and perf_ind <= 1:
    #         applied_err = 1.1 * abstract_err
    #         v_err = Zonotope(0 * applied_err, np.diag(applied_err))
    #         r_all_err = ALK2011HSCC.error_solution(v_err, lin_opt)
    #         r_max = r_delta + cvt2(r_diff, Geometry.TYPE.ZONOTOPE) + r_all_err
    #         true_err, v_err_dyn, v_err_stat = cls.abstract_err(
    #             sys, opt, r_max, r_diff + r_all_err, h, zd, err_stat, t, ind3, zd3
    #         )
    #
    #         # compare linearization error with the maximum allowed error
    #         temp = true_err / applied_err
    #         temp[np.isnan(temp)] = -np.inf
    #         perf_ind_cur = np.max(temp)
    #         perf_ind = np.max(true_err / opt.max_err)
    #         abstract_err = true_err
    #
    #         # exception for set explosion
    #         if np.any(abstract_err > 1e100):
    #             raise Exception("Set Explosion")
    #     # translate reachable sets by linearization point
    #     r_ti += opt.lin_err_x
    #     r_tp += opt.lin_err_x
    #
    #     # compute the reachable set due to the linearization error
    #     r_err = ALK2011HSCC.error_solution(v_err_dyn, lin_opt)
    #
    #     # add the abstraction error to the reachable sets
    #     r_ti += r_err
    #     r_tp += r_err
    #     # determine the best dimension to split the set in order to reduce the
    #     # linearization error
    #     dim_for_split = []
    #     if perf_ind > 1:
    #         raise NotImplementedError  # TODO
    #     # store the linearization error
    #     r_ti = r_ti.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
    #     r_tp = r_tp.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
    #     return r_ti, r_tp, abstract_err, dim_for_split

    @classmethod
    def reach_one_step(cls, dyn: Callable, dims, x, err, opt: Options):
        r_ti, r_tp, abst_err, dims = cls.poly_reach(dyn, dims, x, err, opt)
        if len(dims) <= 0:
            return r_ti, r_tp
        else:
            raise NotImplementedError  # TODO

    # @classmethod
    # def reach_one_step(cls, sys: NonLinSys, r0, err, opt: Options):
    #     r_ti, r_tp = [], []
    #     for i in range(len(r0)):
    #         temp_r_ti, temp_r_tp, abst_err, dims = cls.poly_reach(
    #             sys, r0[i], err[i], opt
    #         )
    #         # check if the initial set has to be split
    #         if len(dims) <= 0:
    #             r_ti.append(temp_r_ti)
    #             r_tp.append(temp_r_tp)
    #         else:
    #             raise NotImplementedError  # TODO
    #
    #     return r_ti, r_tp

    @classmethod
    def reach(cls, dyn: Callable, dims, opts: Options, x: Zonotope):
        m = Model(dyn, dims)
        assert opts.validation(m.dim)

        ri_set, rp_set = [x], []

        next_rp = x

        for step in range(opts.steps_num):
            next_ri, next_rp = cls.reach_one_step(dyn, dims, next_rp, np.zeros(x.shape), opts)
            ri_set.append(next_ri)
            rp_set.append(next_rp)

        return ri_set, rp_set

    @classmethod
    def reach_parallel(cls, dyn: Callable, dims, opts: Options, xs: [Zonotope]):
        def ll_decompose(ll):
            return [list(group) for group in zip(*ll)]

        # init container for storing the results
        rc = []

        partial_reach = partial(cls.reach, dyn, dims, opts)

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(partial_reach, x) for x in xs]

            for future in as_completed(futures):
                try:
                    rc.append(future.result())
                except Exception as e:
                    raise e

            rc = ll_decompose(rc)

            return ll_decompose(rc[0]), ll_decompose(rc[1])

    # @classmethod
    # def reach(cls, sys: NonLinSys, opt: Options):
    #     assert opt.validation(sys.dim)
    #     # init containers for storing the results
    #     time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
    #     ti_set, ti_time, tp_set, tp_time = [], [], [opt.r0], [time_pts[0]]
    #     err = [[np.zeros(r.shape) for r in opt.r0]]
    #
    #     while opt.step_idx < opt.steps_num - 1:
    #         next_ti, next_tp = cls.reach_one_step(sys, tp_set[-1], err[-1], opt)
    #         opt.step_idx += 1
    #         ti_set.append(next_ti)
    #         ti_time.append(time_pts[opt.step_idx - 1: opt.step_idx + 1])
    #         tp_set.append(next_tp)
    #         tp_time.append(time_pts[opt.step_idx])
    #
    #     return ti_set, tp_set, np.vstack(ti_time), np.array(tp_time)

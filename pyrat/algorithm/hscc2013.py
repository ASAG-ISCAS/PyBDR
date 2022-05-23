"""
REF:

Althoff, M. (2013, April). Reachability analysis of nonlinear systems using conservative
 polynomialization and non-convex sets. In Proceedings of the 16th international
 conference on Hybrid systems: computation and control (pp. 173-182).
"""

from __future__ import annotations
import numpy as np
from scipy.special import factorial
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry, Zonotope, Interval
from pyrat.geometry.operation import cvt2
from pyrat.misc import Set, Reachable
from .algorithm import Algorithm
from .hscc2011 import HSCC2011
from .cdc2008 import CDC2008


class HSCC2013:
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
            assert self._validate_inputs()
            assert self._validate_misc(dim)
            return True

    @classmethod
    def pre_stat_err(cls, sys: NonLinSys.Entity, r_delta: Zonotope, opt: Options):
        r_red = cvt2(r_delta, Geometry.TYPE.ZONOTOPE).reduce(
            Zonotope.REDUCE_METHOD, Zonotope.ERROR_ORDER
        )
        # extend teh sets byt the input sets
        u_stat = Zonotope.zero(opt.u.dim)
        z = r_red.card_prod(u_stat)
        z_delta = r_delta.card_prod(u_stat)
        # compute hessian
        h = sys.hessian((opt.lin_err_x, opt.lin_err_u), "numpy")
        t, ind3, zd3 = None, None, None

        # calculate the quadratic map == static second order error
        err_stat_sec = 0.5 * z.quad_map(h)
        err_stat = None
        # third order tensor
        if opt.tensor_order >= 4:
            raise NotImplementedError
        else:
            err_stat = err_stat_sec
            err_stat = err_stat.reduce(
                Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER
            )
        return h, z_delta, err_stat, t, ind3, zd3

    @classmethod
    def abstract_err(cls, sys, opt, r_all, r_diff, h, zd, verr_stat, t, ind3, zd3):
        # compute interval of reachable set
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
            tx, tu = sys.third_order((total_int_x, total_int_u), mod="interval")

            # calculate the lagrange remainder term
            err_dyn_third = Interval.zero(sys.dim)

            # error relates to tx
            for row, col in tx[0]:
                err_dyn_third[row] += dx @ tx[1][row][col] @ dx * dx[col]

            # error relates to tu
            for row, col in tu[0]:
                err_dyn_third[row] += du @ tu[1][row][col] @ du * du[col]

            err_dyn_third *= 1 / 6
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

    @classmethod
    def poly_reach(cls, sys: NonLinSys.Entity, r: Set, opt: Options):
        lin_sys, lin_opt = CDC2008.linearize(sys, r.geometry, opt)
        r_delta = r.geometry - opt.lin_err_x
        r_ti, r_tp = HSCC2011.reach_one_step(lin_sys, r_delta, lin_opt)
        r_diff = HSCC2011.delta_reach(lin_sys, r_delta, lin_opt)
        h, zd, err_stat, t, ind3, zd3 = cls.pre_stat_err(sys, r_delta, opt)
        perf_ind_cur, perf_ind = np.inf, 0
        applied_err, abstract_err, v_err_dyn, v_err_stat = None, r.err, None, None

        while perf_ind_cur > 1 and perf_ind <= 1:
            applied_err = 1.1 * abstract_err
            v_err = Zonotope(0 * applied_err, np.diag(applied_err))
            r_all_err = HSCC2011.error_solution(v_err, lin_opt)
            r_max = r_delta + cvt2(r_diff, Geometry.TYPE.ZONOTOPE) + r_all_err
            true_err, v_err_dyn, v_err_stat = cls.abstract_err(
                sys, opt, r_max, r_diff + r_all_err, h, zd, err_stat, t, ind3, zd3
            )

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
            temp_r_ti, temp_r_tp, dims = cls.poly_reach(sys, r0[i], opt)
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

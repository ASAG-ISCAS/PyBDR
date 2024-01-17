"""
REF:

Althoff, M., Le Guernic, C., & Krogh, B. H. (2011, April). Reachable set computation for
uncertain time-varying linear systems. In Proceedings of the 14th international
conference on Hybrid systems: computation and control (pp. 93-102).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from scipy.linalg import expm
from scipy.special import factorial
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from pybdr.dynamic_system import LinSys
from pybdr.geometry import Geometry, Zonotope, Interval
from pybdr.geometry.operation import cvt2
from .algorithm import Algorithm


class ALK2011HSCC:
    @dataclass
    class Options(Algorithm.Options):
        u_trans: np.ndarray = None
        factors: np.ndarray = None
        taylor_terms: int = 4
        taylor_powers = None
        taylor_err = None
        taylor_f = None
        taylor_input_f = None
        taylor_v = None
        taylor_rv = None
        taylor_r_trans = None
        taylor_input_corr = None
        taylor_ea_int = None
        taylor_ea_t = None
        is_rv = None
        origin_contained = None

        def validation(self, dim: int):
            self._validate_time_related()
            i = np.arange(1, self.taylor_terms + 2)
            self.factors = np.power(self.step, i) / factorial(i)
            # TODO
            return True

    @staticmethod
    def exponential(sys: LinSys, opt: Options):
        xa_abs = abs(sys.xa)
        xa_power = [sys.xa]
        xa_power_abs = [xa_abs]
        m = np.eye(sys.dim, dtype=float)

        for i in range(opt.taylor_terms):
            xa_power.append(xa_power[i] @ sys.xa)
            xa_power_abs.append(xa_power_abs[i] @ xa_abs)
            m += xa_power_abs[i] * opt.factors[i]

        w = expm(xa_abs * opt.step) - m
        w = abs(w)
        e = Interval(-w, w)
        opt.taylor_powers = np.stack(xa_power)
        opt.taylor_err = e

    @classmethod
    def compute_time_interval_err(cls, sys: LinSys, opt: Options):
        # initialize asum
        asum_pos = np.zeros((sys.dim, sys.dim), dtype=float)
        asum_neg = np.zeros((sys.dim, sys.dim), dtype=float)

        for i in range(1, opt.taylor_terms):
            # compute factor
            exp1, exp2 = -(i + 1) / i, -1 / i
            factor = ((i + 1) ** exp1 - (i + 1) ** exp2) * opt.factors[i]
            # init apos, aneg
            apos = np.zeros((sys.dim, sys.dim), dtype=float)
            aneg = np.zeros((sys.dim, sys.dim), dtype=float)
            # obtain positive and negative parts
            pos_ind = opt.taylor_powers[i] > 0
            neg_ind = opt.taylor_powers[i] < 0
            apos[pos_ind] = opt.taylor_powers[i][pos_ind]
            aneg[neg_ind] = opt.taylor_powers[i][neg_ind]
            # compute powers; a factor is always negative
            asum_pos += factor * aneg
            asum_neg += factor * apos
        # instantiate interval matrix
        asum = Interval(asum_neg, asum_pos)
        # write to object structure
        opt.taylor_f = asum + opt.taylor_err

    @classmethod
    def input_time_interval_err(cls, sys: LinSys, opt: Options):
        # initialize asum
        asum_pos = np.zeros((sys.dim, sys.dim), dtype=float)
        asum_neg = np.zeros((sys.dim, sys.dim), dtype=float)

        for i in range(1, opt.taylor_terms + 1):
            # compute factor
            exp1, exp2 = -(i + 1) / i, -1 / i
            factor = ((i + 1) ** exp1 - (i + 1) ** exp2) * opt.factors[i]
            # init apos, aneg
            apos = np.zeros((sys.dim, sys.dim), dtype=float)
            aneg = np.zeros((sys.dim, sys.dim), dtype=float)
            # obtain positive and negative parts
            pos_ind = opt.taylor_powers[i - 1] > 0
            neg_ind = opt.taylor_powers[i - 1] < 0
            apos[pos_ind] = opt.taylor_powers[i - 1][pos_ind]
            aneg[neg_ind] = opt.taylor_powers[i - 1][neg_ind]
            # compute powers; a factor is always negative
            asum_pos += factor * aneg
            asum_neg += factor * apos
        # instantiate interval matrix
        asum = Interval(asum_neg, asum_pos)
        # compute error due to finite taylor series according to interval document
        # "Input Error Bounds in Reachability Analysis"
        e_input = opt.taylor_err * opt.step
        # write to object structure
        opt.taylor_input_f = asum + e_input

    @classmethod
    def input_solution(cls, sys: LinSys, opt: Options):
        v = opt.u if sys.ub is None else sys.ub @ opt.u
        # compute vTrans
        opt.is_rv = True
        if np.all(v.c == 0) and v.z.shape[1] == 1:
            opt.is_rv = False
        v_trans = opt.u_trans if sys.ub is None else sys.ub @ opt.u

        input_solv, input_corr = None, None
        if opt.is_rv:
            # init v_sum
            v_sum = opt.step * v
            a_sum = opt.step * np.eye(sys.dim)
            # compute higher order terms
            for i in range(opt.taylor_terms):
                v_sum += opt.taylor_powers[i] @ (opt.factors[i + 1] * v)
                a_sum += opt.taylor_powers[i] * opt.factors[i + 1]

            # compute overall solution
            input_solv = v_sum + opt.taylor_err * opt.step * v
        else:
            # only a_sum, since v == origin(0)
            a_sum = opt.step * np.eye(sys.dim)
            # compute higher order terms
            for i in range(opt.taylor_terms):
                # compute sum
                a_sum += opt.taylor_powers[i] * opt.factors[i + 1]
            input_solv = Zonotope.zero(sys.dim)

        # compute solution due to constant input
        ea_int = a_sum + opt.taylor_err * opt.step
        input_solv_trans = ea_int * cvt2(v_trans, Geometry.TYPE.ZONOTOPE)
        # compute additional uncertainty if origin is not contained in the input set
        if opt.origin_contained:
            raise NotImplementedError  # TODO
        else:
            # compute inputF
            cls.input_time_interval_err(sys, opt)
            input_corr = opt.taylor_input_f * cvt2(v_trans, Geometry.TYPE.ZONOTOPE)

        # write to object structure
        opt.taylor_v = v
        opt.taylor_rv = input_solv
        opt.taylor_r_trans = input_solv_trans
        opt.taylor_input_corr = input_corr
        opt.taylor_ea_int = ea_int

    @classmethod
    def error_solution(cls, v_dyn: Geometry.Base, opt: Options):
        a_sum = opt.step * v_dyn
        for i in range(opt.taylor_terms):
            a_sum += opt.factors[i + 1] * opt.taylor_powers[i] @ v_dyn

        # get error due to finite taylor series
        f = opt.taylor_err * v_dyn * opt.step
        return a_sum + f

    @classmethod
    def delta_reach(cls, sys: LinSys, r: Geometry.Base, opt: Options):
        rhom_tp_delta = (opt.taylor_ea_t - np.eye(sys.dim)) @ r + opt.taylor_r_trans

        if r.type == Geometry.TYPE.ZONOTOPE:
            o = Zonotope.zero(sys.dim, 0)
            rhom = o.enclose(rhom_tp_delta) + opt.taylor_f * r + opt.taylor_input_corr
        else:
            raise NotImplementedError
        # reduce zonotope
        rhom = rhom.reduce(Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER)
        rv = opt.taylor_rv.reduce(Zonotope.REDUCE_METHOD, Zonotope.INTERMEDIATE_ORDER)

        # final result
        return rhom + rv

    @classmethod
    def reach_one_step(cls, sys: LinSys, r: Zonotope, opt: Options):
        cls.exponential(sys, opt)
        cls.compute_time_interval_err(sys, opt)
        cls.input_solution(sys, opt)
        opt.taylor_ea_t = expm(sys.xa * opt.step)
        r_hom_tp = opt.taylor_ea_t @ r + opt.taylor_r_trans
        r_hom = (
                r.enclose(r_hom_tp)
                + opt.taylor_f * cvt2(r, Geometry.TYPE.ZONOTOPE)
                + opt.taylor_input_corr
        )
        r_hom = r_hom.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
        r_hom_tp = r_hom_tp.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)
        rv = opt.taylor_rv.reduce(Zonotope.REDUCE_METHOD, Zonotope.ORDER)

        return r_hom + rv, r_hom_tp + rv

    @classmethod
    def reach(cls, sys: LinSys, opt: Options, x: Zonotope):
        assert opt.validation(sys.dim)
        ri_set, rp_set = [x], []

        next_rp = x

        for i in range(opt.steps_num):
            next_ri, next_rp = cls.reach_one_step(sys, next_rp, opt)
            ri_set.append(next_ri)
            rp_set.append(next_rp)

        return ri_set, rp_set

    @classmethod
    def reach_parallel(cls, sys: LinSys, opts: Options, xs: [Zonotope]):
        def ll_decompose(ll):
            return [list(group) for group in zip(*ll)]

        with ProcessPoolExecutor() as executor:
            partial_reach = partial(cls.reach, sys, opts)

            futures = [executor.submit(partial_reach, x) for x in xs]

            rc = []

            for future in as_completed(futures):
                try:
                    rc.append(future.result())
                except Exception as exc:
                    raise exc

            rc = ll_decompose(rc)

            return ll_decompose(rc[0]), ll_decompose(rc[1])

    # @classmethod
    # def reach(cls, sys: LinearSystemSimple, opt: Options, x: Zonotope):
    #     assert opt.validation(sys.dim)
    #
    #     ti_set, tp_set = [], [x]
    #
    #     next_tp = x
    #
    #     while opt.step_idx < opt.steps_num - 1:
    #         next_ti, next_tp = cls.reach_one_step(sys, next_tp, opt)
    #         opt.step_idx += 1
    #         ti_set.append(next_ti)
    #         tp_set.append(next_tp)
    #
    #     return ti_set, tp_set

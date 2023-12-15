from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pybdr.dynamic_system import LinSys
from pybdr.geometry import Interval, Geometry, Zonotope
from pybdr.geometry.operation import enclose
from .algorithm import Algorithm
import pybdr.util.functional.auxiliary as aux
from scipy.special import factorial


class ReachLinearZonotope:
    @dataclass
    class Options(Algorithm.Options):
        taylor_terms: int = 4

        def validation(self, dim: int):
            assert self._validate_time_related()

            # TODO

            return True

    @classmethod
    def taylor_cof(cls, n: int):
        i = np.arange(0, n)
        return i, factorial(i)

    @classmethod
    def epsilon(cls, a_inf_norm, t: float, eta: int):
        assert eta > 0
        epsilon = a_inf_norm * t / (eta + 2)
        if not epsilon < 1:
            raise Exception("Epsilon must be less than 1")

        return epsilon

    @classmethod
    def remainder_et(cls, shape, a_norm_inf, t, eta, epsilon, eta_fac):
        return Interval.identity(shape) * ((a_norm_inf * t) ** (eta + 1)) / eta_fac * (1 / 1 - epsilon)

    @classmethod
    def e_at(cls, a_powers, t, eta, et, eta_facs):
        at_powers = a_powers * np.power(t, np.arange(eta + 1))[:, np.newaxis, np.newaxis]
        taylor_sums = ((1 / eta_facs)[:, None, None] * at_powers).sum(axis=0)

        return taylor_sums + et

    @classmethod
    def correction_matrix(cls, a_powers, r, er, ti):
        cof = np.power(ti, -ti / (ti - 1)) - np.power(ti, -1 / (ti - 1))
        inf = cof * np.power(r, ti)
        i = Interval(inf, np.zeros_like(inf))
        taylor_sums = ((i / ti)[:, None, None] * a_powers).sum(axis=0)

        return taylor_sums + er

    @classmethod
    def decompose_u(cls, u: Zonotope):
        return u.c, u - u.c

    @classmethod
    def pr_const(cls, a_inv, e_ar, u_const):
        return a_inv @ (e_ar - np.eye(*e_ar.shape)) @ u_const

    @classmethod
    def r0_const(cls, a_inv, x0: Zonotope, pr_const, f, e_ar, u_const):
        # term 0
        term_0 = enclose(x0, e_ar @ x0 + pr_const, target=Geometry.TYPE.ZONOTOPE)
        # term 1
        term_1 = -1 * pr_const
        # term 2
        term_2 = f @ x0
        # term 3
        term_3 = a_inv @ f @ u_const

        return term_0 + term_1 + term_2 + term_3

    @classmethod
    def v0(cls, u_set: Zonotope, er, r, a_powers, ti, ti_fac):
        # term 0
        term_00 = a_powers * (np.power(r, ti) / ti_fac)[..., None, None]
        term_0 = 0
        for idx in range(term_00.shape[0]):
            this_term_00 = term_00[idx]
            term_0 += this_term_00 @ u_set

        # term 1
        term_1 = er * r @ u_set

        return term_0 + term_1

    @classmethod
    def pr0(cls, pr_const: Interval, v0: Zonotope):
        return pr_const + (v0 - pr_const)

    @classmethod
    def r0(cls, r0_const, pr0, pr_const):
        return r0_const + pr0 + pr_const

    @classmethod
    def pre_compute(cls, sys: LinSys, opt: Options):
        ti, ti_fac = cls.taylor_cof(opt.taylor_terms + 2)
        ap = aux.mat_powers_2d(sys.xa, opt.taylor_terms)
        a_norm_inf = np.linalg.norm(sys.xa, np.inf)
        a_inv = np.linalg.pinv(sys.xa)
        epsilon = cls.epsilon(a_norm_inf, opt.step, opt.taylor_terms)
        er = cls.remainder_et(sys.xa.shape, a_norm_inf, opt.step, opt.taylor_terms, epsilon, ti_fac[-1])
        e_ar = cls.e_at(ap, opt.step, opt.taylor_terms, er, ti_fac[:-1])
        u_const, u_set = cls.decompose_u(opt.u)
        F = cls.correction_matrix(ap[2:], opt.step, er, ti[2:-1])

        pr_const = cls.pr_const(a_inv, e_ar, u_const)
        r0_const = cls.r0_const(a_inv, opt.r0, pr_const, F, e_ar, u_const)
        v0 = cls.v0(u_set, er, opt.step, ap, ti[1:], ti_fac[1:])
        pr0 = cls.pr0(pr_const, v0)
        r0 = cls.r0(r0_const, pr0, pr_const)

        return e_ar, pr0, pr_const, v0, r0

    @classmethod
    def reach_one_step(cls, e_ar, pr_pre, r_const_pre, v_pre):
        r_const_next = e_ar @ r_const_pre
        v_next = e_ar @ v_pre
        pr_next = pr_pre + v_next
        r_next = r_const_next + pr_next
        return r_next, pr_next, r_const_next, v_next

    @classmethod
    def reach(cls, sys: LinSys, opt: Options):
        assert opt.validation(sys.dim)

        tp_set = [opt.r0]

        e_ar, pr, r_const, v, r = cls.pre_compute(sys, opt)
        tp_set.append(r)

        for k in range(opt.steps_num):
            r, pr, r_const, v = cls.reach_one_step(e_ar, pr, r_const, v)
            tp_set.append(r)

        return None, tp_set, None, None

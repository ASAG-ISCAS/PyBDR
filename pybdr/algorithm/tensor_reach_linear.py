from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pybdr.dynamic_system import LinSys
from pybdr.geometry import Interval, Geometry
from pybdr.geometry.operation import enclose
from .algorithm import Algorithm
import pybdr.util.functional.auxiliary as aux
from scipy.linalg import expm
from scipy.special import factorial

"""
Reachability Analysis and its Application to the Safety Assessment of Autonomous Cars
3.2 Linear Continuous Systems, Algorithm 3
"""


class IntervalTensorReachLinear:
    @dataclass
    class Options(Algorithm.Options):
        taylor_terms: int = 4

        def validation(self, dim: int):
            assert self._validate_time_related()
            assert self.taylor_terms >= 2  # at least 2
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
        if not epsilon < 1:  # check if this epsilon is valid or not
            raise Exception("Epsilon must be less than 1")
        return epsilon

    @classmethod
    def remainder_et(cls, shape, a_norm_inf: float, t: float, eta: int, epsilon: float, eta_fac: float):
        return Interval.identity(shape) * ((a_norm_inf * t) ** (eta + 1)) / eta_fac * (1 / 1 - epsilon)

    @classmethod
    def e_at(cls, a_powers: np.ndarray, t: float, eta: int, et: Interval, eta_facs: np.ndarray):
        """
        compute e^{a*t}, where a is the state matrix of the linear system and t is the step of the computation
        @param eta_facs:
        @param et:
        @param eta: number of taylor terms for summing
        @param a: state matrix of the linear system
        @param t: step of the computation
        @return: over-approximation of the e^{A*t}
        """
        # compute epsilon
        # assert a.ndim == 2  # state matrix must be 2d
        # compute sum of taylor terms
        # m = np.eye(*a.shape)
        # a_powers = [m]
        # for idx in range(eta):
        #     a_powers.append(a_powers[-1] @ (a * t))
        # a_powers = np.stack(a_powers)
        at_powers = a_powers * np.power(t, np.arange(eta + 1))[:, np.newaxis, np.newaxis]
        taylor_sums = ((1 / eta_facs)[:, None, None] * at_powers).sum(axis=0)

        return taylor_sums + et

    @classmethod
    def correction_matrix(cls,
                          a_powers: np.ndarray,
                          r: float,
                          eta: int,
                          er: Interval,
                          ti: np.ndarray,
                          ti_fac: np.ndarray):
        """
        compute the correction matrix F which determines the enlargement
        @param er:
        @param a:
        @param r:
        @param eta:
        @return:
        """
        # compute sum of taylor terms
        cof = np.power(ti, -ti / (ti - 1)) - np.power(ti, -1 / (ti - 1))
        inf = cof * np.power(r, ti)
        i = Interval(inf, np.zeros_like(inf))
        taylor_sums = ((i / ti)[:, None, None] * a_powers).sum(axis=0)

        return taylor_sums + er

    @classmethod
    def decompose_u(cls, u: Interval):
        """
        Decompose the input U into constant u and U which contains origin
        @param u: input control set U
        @return: u_const and u_set
        """
        u_const = np.zeros(u.shape)
        if u.contains(u_const):
            return u_const, u
        u_const = u.c
        return u_const, u - u_const

    @classmethod
    def pr_const(cls, a_inv: np.ndarray, e_ar: Interval, u_const):
        return a_inv @ (e_ar - np.eye(*e_ar.shape)) @ u_const

    @classmethod
    def r0_const(cls,
                 a_inv: np.ndarray,
                 x0: Interval,
                 pr_const: Interval,
                 f: Interval,
                 e_ar: Interval,
                 u_const: np.ndarray):
        # term 0
        term_00 = Interval.squeeze(e_ar[None, ...] @ x0[:, ..., None], axis=-1) + pr_const
        term_0 = enclose(x0, term_00, Geometry.TYPE.INTERVAL)
        # term 1
        term_1 = -1 * pr_const
        # term 2
        term_2 = Interval.squeeze(f[None, ...] @ x0[:, ..., None], axis=-1)
        # term 3
        term_3 = Interval.squeeze(a_inv @ f @ u_const[..., None], axis=-1)

        return term_0 + term_1 + term_2 + term_3

    @classmethod
    def v0(cls, u_set: Interval, er: Interval, r: float, eta: int, a_powers: np.ndarray, ti: np.ndarray,
           ti_fac: np.ndarray):
        # term 0
        term_00 = a_powers * (np.power(r, ti) / ti_fac)[..., None, None]
        term_0 = Interval.squeeze(term_00 @ u_set[None, :, None], axis=-1).sum(axis=0)
        # term 1
        term_1 = er * r @ u_set

        return term_0 + term_1

    @classmethod
    def pr0(cls, pr_const: Interval, v0: Interval):
        return pr_const + (v0 - pr_const)

    @classmethod
    def r0(cls, r0_const: Interval, pr0: Interval, pr_const: Interval):
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
        F = cls.correction_matrix(ap[2:], opt.step, opt.taylor_terms, er, ti[2:-1], ti_fac[2:-1])

        # inhomogeneous reachable set due to constant input
        pr_const = cls.pr_const(a_inv, e_ar, u_const)
        r0_const = cls.r0_const(a_inv, opt.r0, pr_const, F, e_ar, u_const)
        v0 = cls.v0(u_set, er, opt.step, opt.taylor_terms, ap, ti[1:], ti_fac[1:])
        pr0 = cls.pr0(pr_const, v0)
        r0 = cls.r0(r0_const, pr0, pr_const)

        return e_ar, pr0, r0_const, v0, r0

    @classmethod
    def reach_one_step(cls, e_ar: Interval, pr_pre: Interval, r_const_pre: Interval, v_pre: Interval):
        r_const_next = Interval.squeeze(e_ar[None, ...] @ r_const_pre[:, ..., None], axis=-1)
        v_next = e_ar @ v_pre
        pr_next = pr_pre + v_next
        r_next = r_const_next + pr_next
        return r_next, pr_next, r_const_next, v_next

    @classmethod
    def reach(cls, sys: LinSys, opt: Options):
        """
        compute reachable set for given linear system with specified input from initial sets in interval tensor form
        @param sys: given linear system
        @param opt: options
        @return:
        """
        assert opt.validation(sys.dim)  # validate the settings
        # init container for storing the results
        time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
        ti_set, ti_time, tp_set, tp_time = [], [], [opt.r0], [time_pts[0]]

        e_ar, pr, r_const, v, r = cls.pre_compute(sys, opt)
        tp_set.append(r)

        # return None, tp_set, None, None

        for k in range(opt.steps_num):
            r, pr, r_const, v = cls.reach_one_step(e_ar, pr, r_const, v)
            tp_set.append(r)

        return None, tp_set, None, None

        return ti_set, tp_set, np.vstack(ti_time), np.array(tp_time)

from __future__ import annotations

import numpy as np
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.geometry import Interval, Geometry, Zonotope
from pybdr.geometry.operation import enclose, cvt2
from dataclasses import dataclass
from scipy.special import factorial


class ReachLinearZonoAlgo3:
    @dataclass
    class Settings:
        t_end: float = 0
        step: float = 0
        eta: int = 4  # number of taylor terms in approximating using taylor series
        x0: Zonotope = None
        u: Zonotope = None

        def __init__(self):
            self._num_steps = 0

        def validation(self):
            assert self.t_end >= self.step >= 0
            assert self.eta >= 3
            self._num_steps = round(self.t_end / self.step)
            return True

        @property
        def num_steps(self):
            return self._num_steps

    @classmethod
    def compute_epsilon(cls, a, t, eta):
        assert eta > 0
        a_inf_norm = np.linalg.norm(a, np.inf)
        epsilon = (a_inf_norm * t) / (eta + 2)
        if epsilon >= 1:
            raise Exception("Epsilon must be less than 1")
        return epsilon

    @classmethod
    def compute_et(cls, a, t, eta, epsilon):
        a_norm = np.linalg.norm(a, np.inf)
        cof = ((a_norm * t) ** (eta + 1) / factorial(eta + 1)) * 1 / (1 - epsilon)
        return Interval.identity(a.shape) * cof

    @classmethod
    def compute_e_ar(cls, eta, r, a):
        # compute epsilon
        epsilon = cls.compute_epsilon(a, r, eta)
        # compute sums of taylor terms
        taylor_sums = 0
        for i in range(eta + 1):
            taylor_sums += 1 / factorial(i) * np.linalg.matrix_power(a * r, i)

        er = cls.compute_et(a, r, eta, epsilon)

        return taylor_sums + er, er

    @classmethod
    def compute_f(cls, eta, a, er, r):
        # compute taylor sums
        taylor_sums = 0

        for i in range(2, eta + 1):
            cof = np.linalg.matrix_power(a, i) / factorial(i)
            box_inf = (np.power(i, -i / (i - 1)) - np.power(i, -1 / (i - 1))) * np.power(r, i)
            box = Interval(box_inf, 0)
            taylor_sums += box * cof

        return taylor_sums + er

    @classmethod
    def compute_v(cls, a, eta, u, r, er):
        # compute taylor sums
        taylor_sums = 0
        for i in range(0, eta + 1):
            cof = np.linalg.matrix_power(a, i) * np.power(r, i + 1) / factorial(i + 1)
            taylor_sums += cof @ u

        return taylor_sums + er * r @ u

    @classmethod
    def decompose_u(cls, u: Zonotope):
        """
        decompose u into u_const and u_set so that u_set contains origin
        @param u:
        @return:
        """
        return u.c, u - u.c

    @classmethod
    def compute_pr_const(cls, a_inv, e_ar, u_const):
        return a_inv @ (e_ar - np.eye(*e_ar.shape)) @ u_const

    @classmethod
    def compute_pr(cls, prc, v):
        return prc + cvt2(v - prc, Geometry.TYPE.INTERVAL)

    @classmethod
    def compute_r_const(cls, e_ar, prc, f, x0, a_inv, uc):
        # term 00
        term_00 = enclose(x0, e_ar @ x0 + prc, Geometry.TYPE.ZONOTOPE)
        # term 01
        term_01 = -1 * prc
        # term 02
        term_02 = f @ x0
        # term 03
        term_03 = a_inv @ f @ uc

        return term_00 + term_01 + term_02 + term_03

    @classmethod
    def compute_r(cls, rc, pr, prc):
        return rc + pr + prc

    @classmethod
    def pre_compute(cls, lin_sys: LinearSystemSimple, opts: Settings):
        a_inv = np.linalg.pinv(lin_sys.xa)
        e_ar, er = cls.compute_e_ar(opts.eta, opts.step, lin_sys.xa)
        f = cls.compute_f(opts.eta, lin_sys.xa, er, opts.step)
        uc, us = cls.decompose_u(opts.u)
        prc = cls.compute_pr_const(a_inv, e_ar, uc)
        rc = cls.compute_r_const(e_ar, prc, f, opts.x0, a_inv, uc)
        v = cls.compute_v(lin_sys.xa, opts.eta, opts.u, opts.step, er)
        pr = cls.compute_pr(prc, v)
        r = cls.compute_r(rc, pr, prc)

        return r, e_ar, rc, v, pr

    @classmethod
    def reach_one_step(cls, e_ar, rc_cur, v_cur, pr_cur):
        rc_next = e_ar @ rc_cur
        v_next = e_ar @ v_cur
        pr_next = pr_cur + cvt2(v_next, Geometry.TYPE.INTERVAL)
        r_next = rc_next + pr_next

        return r_next, rc_next, v_next, pr_next

    @classmethod
    def reach(cls, lin_sys: LinearSystemSimple, opts: Settings):
        assert opts.validation()

        r0, e_ar, rc_cur, v_cur, pr_cur = cls.pre_compute(lin_sys, opts)

        ri = [opts.x0, r0]

        for k in range(opts.num_steps):
            r_cur, rc_cur, v_cur, pr_cur = cls.reach_one_step(e_ar, rc_cur, v_cur, pr_cur)
            ri.append(r_cur)

        return None, ri, None, None

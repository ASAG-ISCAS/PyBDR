from __future__ import annotations

import numpy as np
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.geometry import Interval, Geometry, Zonotope
from pybdr.geometry.operation import enclose, cvt2
from dataclasses import dataclass
from scipy.special import factorial


class ReachLinearZonoAlgo2:
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
    def compute_v(cls, a, eta, u, r, er):
        # compute taylor sums
        taylor_sums = 0
        for i in range(0, eta + 1):
            cof = np.linalg.matrix_power(a, i) * np.power(r, i + 1) / factorial(i + 1)
            taylor_sums += cof @ u

        return taylor_sums + er * r @ u

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
    def compute_hr(cls, e_ar, x0, f):
        return enclose(x0, e_ar @ x0, Geometry.TYPE.ZONOTOPE) + f @ x0

    @classmethod
    def pre_compute(cls, lin_sys: LinearSystemSimple, opts: Settings):
        e_ar, er = cls.compute_e_ar(opts.eta, opts.step, lin_sys.xa)
        f = cls.compute_f(opts.eta, lin_sys.xa, er, opts.step)
        v = cls.compute_v(lin_sys.xa, opts.eta, opts.u, opts.step, er)
        pr = cvt2(v, Geometry.TYPE.INTERVAL)
        hr = cls.compute_hr(e_ar, opts.x0, f)
        r = hr + pr

        return r, hr, pr, v, e_ar

    @classmethod
    def reach_one_step(cls, e_ar, hr_cur, v_cur, pr_cur):
        hr_next = e_ar @ hr_cur
        v_next = e_ar @ v_cur
        pr_next = pr_cur + cvt2(v_next, Geometry.TYPE.INTERVAL)
        r_next = hr_next + pr_next

        return r_next, hr_next, v_next, pr_next

    @classmethod
    def reach(cls, lin_sys: LinearSystemSimple, opts: Settings):
        assert opts.validation()

        r0, hr_cur, pr_cur, v_cur, e_ar = cls.pre_compute(lin_sys, opts)

        ri = [opts.x0, r0]

        for k in range(opts.num_steps):
            r_cur, hr_cur, v_cur, pr_cur = cls.reach_one_step(e_ar, hr_cur, v_cur, pr_cur)
            ri.append(r_cur)

        return None, ri, None, None

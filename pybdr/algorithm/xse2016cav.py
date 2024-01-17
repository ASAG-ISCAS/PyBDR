"""
REF:

Xue, B., She, Z., & Easwaran, A. (2016, July). Under-approximating backward reachable
sets by polytopes. In International Conference on Computer Aided Verification
(pp. 457-476). Springer, Cham.
"""

from __future__ import annotations
from dataclasses import dataclass
import cvxpy as cp
import numpy as np
from typing import Callable
from functools import partial
from pybdr.geometry import Geometry, Polytope, Zonotope
from pybdr.geometry.operation import cvt2, boundary
from pybdr.model import Model
from .algorithm import Algorithm
from .asb2008cdc import ASB2008CDC

np.seterr(divide="ignore", invalid="ignore")


def _rev_dyn(dyn, x, u):
    return -1 * dyn(x, u)


class XSE2016CAV:
    @dataclass
    class Options(Algorithm.Options):
        epsilon_m: float = np.inf  # for boundary sampling
        epsilon: float = np.inf  # for backward verification

        def validation(self, dim: int):
            assert self._validate_time_related()
            assert not np.isinf(self.epsilon) and self.epsilon >= 0
            assert not np.isinf(self.epsilon_m) and self.epsilon_m > 0
            self.step_idx = self.steps_num - 1  # index from 0
            return True

    @classmethod
    def boundary_back(cls, dyn: Callable, dims, u, epsilon, opt: ASB2008CDC.Options):

        rev_dyn = partial(_rev_dyn, dyn)

        m = Model(rev_dyn, dims)
        m.reverse()
        bounds = boundary(u, epsilon, Geometry.TYPE.ZONOTOPE)
        _, rp = ASB2008CDC.reach_parallel(m.f, dims, opt, bounds)
        return [cvt2(zono, Geometry.TYPE.INTERVAL) for zono in rp[-1]]

    @classmethod
    def polytope(cls, omega):
        # get vertices of these input geometry objects
        pts = np.concatenate([interval.vertices for interval in omega], axis=0)
        # get polytope from these points
        return cvt2(pts, Geometry.TYPE.POLYTOPE)

    @classmethod
    def contraction(cls, omega, o):
        num_box = len(omega)
        bj = []
        for i in range(num_box):
            x = cp.Variable(o.shape + 1)
            c = np.zeros(o.shape + 1)
            c[-1] = 1

            constraints = []
            a = np.zeros((o.a.shape[0], x.shape[0]))
            a[:, :-1] = o.a
            a[:, -1] = -1
            constraints.append(a @ x - o.b <= 0)
            lb = np.zeros(omega[i].inf.shape[0] + 1)
            lb[:-1] = omega[i].inf
            lb[-1] = -1e15
            ub = np.zeros(omega[i].sup.shape[0] + 1)
            ub[:-1] = omega[i].sup
            ub[-1] = 0
            constraints.append(lb <= x)
            constraints.append(x <= ub)

            cost = c @ x
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.GLPK)
            assert prob.status == "optimal"  # ensure valid solution for LP problem
            bj.append(x.value[-1])
        bu = np.min(bj)
        return Polytope(o.a, o.b + bu), bu

    @classmethod
    def get_d(cls, o: Polytope):  # polytope before contraction
        x = cp.Variable(o.shape + 1)
        c = np.zeros(o.shape + 1)
        c[-1] = 1

        constraints = []
        a = np.zeros((o.a.shape[0], x.shape[0]))
        a[:, :-1] = o.a
        a[:, -1] = -1
        constraints.append(a @ x - o.b <= 0)

        cost = c @ x
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.GLPK)
        assert prob.status == "optimal"  # ensure valid solution for LP
        return x.value[-1]  # which is d

    @classmethod
    def verification(cls, dyn, dims, o, u_back, bu, epsilon, opt: ASB2008CDC.Options):
        r0 = Zonotope(u_back.c, np.eye(u_back.c.shape[0]) * 0.1)
        _, rp = ASB2008CDC.reach(dyn, dims, opt, r0)
        sx = rp[-1]
        is_inside = sx in o
        d = cls.get_d(o)
        if abs(bu / d) > epsilon or not is_inside:
            return False
        return True

    @classmethod
    def one_step_backward(cls, dyn: Callable, dims, u, opt: Options, opt_back: ASB2008CDC.Options):

        omega = cls.boundary_back(dyn, dims, u, opt.epsilon_m, opt_back)
        o = cls.polytope(omega)
        u_back, bu = cls.contraction(omega, o)
        if not cls.verification(dyn, dims, o, u_back, bu, opt.epsilon, opt_back):
            return None, False
        return u_back, True

    @classmethod
    def reach(cls, dyn: Callable, dims, opt: Options, opt_back: ASB2008CDC.Options):
        sys = Model(dyn, dims)
        assert opt.validation(sys.dim)
        assert opt_back.validation(sys.dim)
        tp_set, tp_time = [], []
        time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
        u = opt.r0
        tp_set.append(u)
        tp_time.append(time_pts[-1])

        # loop over all backward steps
        while opt.step_idx >= 0:
            u, is_valid = cls.one_step_backward(dyn, dims, u, opt, opt_back)
            opt.step_idx -= 1
            if is_valid:
                tp_set.append(u)
                tp_time.append(time_pts[-opt.step_idx - 1])
            else:
                break

        return tp_set, tp_time

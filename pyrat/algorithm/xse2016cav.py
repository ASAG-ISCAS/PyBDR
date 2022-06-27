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
from scipy.linalg import block_diag

from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry, Polytope, Zonotope
from pyrat.geometry.operation import cvt2, boundary
from pyrat.misc import Reachable
from .algorithm import Algorithm
from .asb2008cdc import ASB2008CDC


class CAV2016:
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
    def boundary_back(cls, sys: NonLinSys, u, epsilon, opt: ASB2008CDC.Options):
        bounds = boundary(u, epsilon, Geometry.TYPE.ZONOTOPE)
        r0 = [cvt2(bd, Geometry.TYPE.ZONOTOPE) for bd in bounds]
        opt.r0 = r0
        rs = ASB2008CDC.reach(sys, opt)
        return [cvt2(zono.geometry, Geometry.TYPE.INTERVAL) for zono in rs.tps[-1]]

    @classmethod
    def polytope(cls, omega):
        # get vertices of these input geometry objects
        pts = np.concatenate([interval.vertices for interval in omega], axis=0)
        # get polytope from these points
        return cvt2(pts, Geometry.TYPE.POLYTOPE)

    @classmethod
    def contraction(cls, omega, o, opt: Options):
        num_box = len(omega)
        bj = []
        for i in range(num_box):
            x = cp.Variable(o.dim + 1)
            c = np.zeros(o.dim + 1)
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
            prob.solve()
            assert prob.status == "optimal"  # ensure valid solution for LP problem
            bj.append(x.value[-1])
        bu = np.min(bj)
        return Polytope(o.a, o.b + bu), bu

    @classmethod
    def contraction_try(cls, omega, o, opt: Options):
        # do linear programming simultaneously
        num_box = len(omega)
        x = cp.Variable(num_box * o.dim + num_box)
        constraints = []
        # add constraints Ax+C<=B
        a = np.zeros((o.a.shape[0] * num_box, x.shape[0]))
        temp0 = a[:, :-num_box]
        temp1 = block_diag(*[o.a for i in range(num_box)])
        print(temp1.shape)
        print(temp0.shape)
        a[:, :-num_box] = block_diag(*[o.a for i in range(num_box)])
        constraints.append(a @ x <= np.repeat(o.b, num_box))
        # add constraints x in Ij
        bd = np.full((x.shape[0], 2), -np.inf)
        bd[:-num_box, 0] = np.concatenate([interval.inf for interval in omega])
        bd[:-num_box, 1] = np.concatenate([interval.sup for interval in omega])
        bd[-num_box:1] = 0
        constraints.append(bd[:, 0] <= x)
        constraints.append(x <= bd[:, 1])
        assert np.all(bd[:, 0] <= bd[:, 1])
        # minimize c'x with x=[xj0 xj1 ... b0 b1 b2 ... b_{mk}] == sum of bj
        c = np.zeros(x.shape[0])
        c[-num_box:] = 1
        cost = c @ x
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.MOSEK, verbose=True)
        # check the linear programming

        exit(False)
        # TODO
        raise NotImplementedError

    @classmethod
    def get_d(cls, o: Polytope):  # polytope before contraction
        x = cp.Variable(o.dim + 1)
        c = np.zeros(o.dim + 1)
        c[-1] = 1

        constraints = []
        a = np.zeros((o.a.shape[0], x.shape[0]))
        a[:, :-1] = o.a
        a[:, -1] = -1
        constraints.append(a @ x - o.b <= 0)

        cost = c @ x
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        assert prob.status == "optimal"  # ensure valid solution for LP
        return x.value[-1]  # which is d

    @classmethod
    def verification(cls, o, u_back, sys, bu, epsilon, opt: ASB2008CDC.Options):
        r0 = Zonotope(u_back.c, np.eye(u_back.c.shape[0]) * 0.1)
        opt.r0 = [r0]
        rs = ASB2008CDC.reach(sys, opt)
        sx = rs.tps[-1][-1].geometry
        is_inside = sx in o
        d = cls.get_d(o)
        if abs(bu / d) > epsilon or not is_inside:
            return False
        return True

    @classmethod
    def one_step_backward(cls, u, sys, opt: Options, opt_back: ASB2008CDC.Options):
        sys.reverse()  # reverse the system for backward computation
        omega = cls.boundary_back(sys, u, opt.epsilon_m, opt_back)
        o = cls.polytope(omega)
        u_back, bu = cls.contraction(omega, o, opt)
        sys.reverse()  # reverse the system for forward computation
        if not cls.verification(o, u_back, sys, bu, opt.epsilon, opt_back):
            return None, False
        return u_back, True

    @classmethod
    def reach(cls, sys: NonLinSys, opt: Options, opt_back: ASB2008CDC.Options):
        assert opt.validation(sys.dim)
        assert opt_back.validation(sys.dim)
        tp_set, tp_time = [], []
        time_pts = np.linspace(opt.t_start, opt.t_end, opt.steps_num)
        u = opt.r0
        tp_set.append(u)
        tp_time.append(time_pts[-1])

        # loop over all backward steps
        while opt.step_idx >= 0:
            u, is_valid = cls.one_step_backward(u, sys, opt, opt_back)
            opt.step_idx -= 1
            if is_valid:
                tp_set.append(u)
                tp_time.append(time_pts[-opt.step_idx - 1])
            else:
                break

        return Reachable.Result([], tp_set)

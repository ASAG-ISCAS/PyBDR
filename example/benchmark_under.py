from __future__ import annotations

import numpy as np

np.seterr(divide="ignore", invalid="ignore")

from pybdr.geometry import Geometry, Zonotope, Interval
from pybdr.geometry.operation import cvt2, boundary
from pybdr.algorithm import ASB2008CDC, XSE2016CAV
from pybdr.dynamic_system import NonLinSys
from pybdr.model import Model, synchronous_machine
from pybdr.util.visualization import plot

if __name__ == "__main__":
    # init system
    system = NonLinSys(Model(synchronous_machine, [2, 1]))

    epsilon = 0.1
    opt = ASB2008CDC.Options()
    opt.t_end = 1
    opt.step = 0.05
    opt.r0 = Interval([-0.1, 0.1], [2.9, 3.1])
    opt.tensor_order = 2
    opt.taylor_terms = 4
    opt.u = Zonotope.zero(1, 1)
    opt.u_trans = np.zeros(1)

    # reach
    assert opt.validation(system.dim)
    tp_set = [opt.r0]
    opt.r0 = boundary(opt.r0, epsilon, Geometry.TYPE.ZONOTOPE)

    _, tp, _, _ = ASB2008CDC.reach(system, opt)

    vis_idx = [int(len(tp) / 2) - 1]

    for this_idx in vis_idx:
        omega = [cvt2(zono, Geometry.TYPE.INTERVAL) for zono in tp[this_idx]]
        o = XSE2016CAV.polytope(omega)
        p, _ = XSE2016CAV.contraction(omega, o)
        tp_set.append(p)
        tp_set.append(o)
        tp_set += omega

    plot(tp_set, [0, 1])

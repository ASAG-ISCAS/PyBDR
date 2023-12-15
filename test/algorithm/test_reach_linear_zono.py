import numpy as np
from pybdr.geometry import Zonotope, Geometry, Interval
from pybdr.geometry.operation import cvt2
from pybdr.dynamic_system import LinSys
from pybdr.algorithm import ReachLinearZonotope
from pybdr.util.visualization import plot


def test_reach_linear_zono_case_00():
    # init settings
    xa = np.array([[-1, -4], [4, -1]])
    ub = np.array([[1], [1]])
    lin_sys = LinSys(xa=xa)
    opt = ReachLinearZonotope.Options()
    opt.u = cvt2(ub @ Interval(-0.1, 0.1), Geometry.TYPE.ZONOTOPE)
    opt.taylor_terms = 4
    opt.t_end = 0.04
    opt.step = 0.01

    r0 = Interval([0.9, 0.9], [1.1, 1.1])
    opt.r0 = cvt2(r0, Geometry.TYPE.ZONOTOPE)

    _, tp_set, _, _ = ReachLinearZonotope.reach(lin_sys, opt)

    # vis
    plot(tp_set, [0, 1])

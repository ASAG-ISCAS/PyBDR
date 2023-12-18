import numpy as np
from pybdr.geometry import Geometry, Interval
from pybdr.geometry.operation import boundary
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.algorithm import ReachLinearIntervalAlgo1
from pybdr.util.visualization import plot


def test_reach_linear_interval_algo1_case_00():
    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearIntervalAlgo1.Settings()
    opts.t_end = 5
    opts.step = 0.01
    opts.eta = 4
    opts.x0 = Interval([0.99, 0.99], [1.01, 1.01])

    _, ri, _, _ = ReachLinearIntervalAlgo1.reach(lin_sys, opts)

    plot(ri, [0, 1])


def test_reach_linear_interval_algo1_case_01():
    """
    this test demonstrates that using interval computing reachable sets is pretty rougher than zonotope
    """
    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearIntervalAlgo1.Settings()
    opts.t_end = 5
    opts.step = 0.04
    opts.eta = 4

    x0 = Interval([0.9, 0.9], [1.1, 1.1])

    bounds = boundary(x0, 0.01, Geometry.TYPE.INTERVAL)

    ri_all = []

    for bound in bounds:
        opts.x0 = bound
        _, ri, _, _ = ReachLinearIntervalAlgo1.reach(lin_sys, opts)
        ri_all.append(ri)

    from itertools import chain

    ri_all = list(chain.from_iterable(ri_all))

    plot(ri_all, [0, 1])

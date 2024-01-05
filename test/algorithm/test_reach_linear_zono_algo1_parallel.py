import numpy as np
from pybdr.geometry import Zonotope, Geometry, Interval
from pybdr.geometry.operation import cvt2, boundary
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.algorithm import ReachLinearZonoAlgo1Parallel
from pybdr.util.visualization import plot
from pybdr.util.functional import performance_counter, performance_counter_start


def test_case_00():
    time_tag = performance_counter_start()
    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearZonoAlgo1Parallel.Settings()
    opts.t_end = 5
    opts.step = 0.04
    opts.eta = 4
    x0 = Interval([0.9, 0.9], [1.1, 1.1])

    x0_bounds = boundary(x0, 0.0001, Geometry.TYPE.ZONOTOPE)

    print(len(x0_bounds))

    _, ri, _, _ = ReachLinearZonoAlgo1Parallel.reach_parallel(lin_sys, opts, x0_bounds)

    performance_counter(time_tag, 'reach_linear_zono_algo1_parallel')

    # plot([*ri, x0], [0, 1])


def test_case_01():
    time_tag = performance_counter_start()
    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearZonoAlgo1Parallel.Settings()
    opts.t_end = 5
    opts.step = 0.04
    opts.eta = 4
    x0 = Interval([0.9, 0.9], [1.1, 1.1])

    x0_bounds = boundary(x0, 0.01, Geometry.TYPE.ZONOTOPE)

    print(len(x0_bounds))

    _, ri, _, _ = ReachLinearZonoAlgo1Parallel.reach_parallel(lin_sys, opts, [x0_bounds[0]])

    performance_counter(time_tag, 'reach_linear_zono_algo1_parallel')

    print(len(ri))

    # plot([*ri, x0], [0, 1])

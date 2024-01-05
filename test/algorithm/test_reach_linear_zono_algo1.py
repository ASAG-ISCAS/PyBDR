import numpy as np
from pybdr.geometry import Zonotope, Geometry, Interval
from pybdr.geometry.operation import cvt2
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.algorithm import ReachLinearZonoAlgo1
from pybdr.util.visualization import plot

"""
Reachable set for linear time-invariant systems without input
"""


def test_reach_linear_zono_algo1_case_00():
    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearZonoAlgo1.Settings()
    opts.t_end = 5
    opts.step = 0.04
    opts.eta = 4
    x0 = cvt2(Interval([0.9, 0.9], [1.1, 1.1]), Geometry.TYPE.ZONOTOPE)

    _, ri, _, _ = ReachLinearZonoAlgo1.reach(lin_sys, opts, x0)

    plot(ri, [0, 1])


def test_reach_linear_zono_algo1_case_01():
    xa = np.array([[-1, -4, 0, 0, 0], [4, - 1, 0, 0, 0], [0, 0, -3, 1, 0], [0, 0, -1, -3, 0], [0, 0, 0, 0, -2]])
    ub = np.eye(5)

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearZonoAlgo1.Settings()
    opts.t_end = 5
    opts.step = 0.04
    opts.eta = 4
    x0 = cvt2(Interval(0.9 * np.ones(5), 1.1 * np.ones(5)), Geometry.TYPE.ZONOTOPE)

    _, ri, _, _ = ReachLinearZonoAlgo1.reach(lin_sys, opts, x0)

    plot(ri, [0, 1])
    plot(ri, [1, 2])
    plot(ri, [2, 3])
    plot(ri, [3, 4])


def test_reach_linear_zono_algo1_case_02():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pybdr.geometry.operation import boundary

    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearZonoAlgo1.Settings()
    opts.t_end = 5
    opts.step = 0.04
    opts.eta = 4

    x0 = Interval([0.9, 0.9], [1.1, 1.1])
    x0_bounds = boundary(x0, 0.1, Geometry.TYPE.ZONOTOPE)

    def parallel_reach(x):
        return ReachLinearZonoAlgo1.reach(lin_sys, opts, x)

    result = None

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(parallel_reach, bound) for bound in x0_bounds]

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                print(exc)

    print(type(result))
    print(len(result))

    # plot(ri, [0, 1])

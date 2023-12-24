from numba import jit
from numba.typed import List
import numpy as np
from pybdr.geometry import Zonotope, Geometry, Interval
from pybdr.geometry.operation import cvt2, boundary
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.algorithm import ReachLinearZonoAlgo1
from pybdr.util.visualization import plot


def test_case_00_without_parallelization():
    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    lin_sys = LinearSystemSimple(xa, ub)
    opts = ReachLinearZonoAlgo1.Settings()
    opts.t_end = 0.5
    opts.step = 0.01
    opts.eta = 4
    opts.x0 = cvt2(Interval([0.9, 0.9], [1.1, 1.1]), Geometry.TYPE.ZONOTOPE)

    _, ri, _, _ = ReachLinearZonoAlgo1.reach(lin_sys, opts)

    plot(ri, [0, 1])


def test_case_00_with_parallelization():
    """
    https://numba.readthedocs.io/en/stable/user/5minguide.html
    https://numba.readthedocs.io/en/stable/user/parallel.html
    @return:
    """

    xa = [[-1, -4], [4, -1]]
    ub = np.array([[1], [1]])

    # lin_sys = LinearSystemSimple(xa, ub)
    # opts = ReachLinearZonoAlgo1.Settings()
    # opts.t_end = 0.5
    # opts.step = 0.01
    # opts.eta = 4
    x0 = Interval([0.9, 0.9], [1.1, 1.1])
    x0_bounds = boundary(x0, 0.05, Geometry.TYPE.INTERVAL)
    x0_bounds = [cvt2(this_bound, Geometry.TYPE.ZONOTOPE) for this_bound in x0_bounds]

    ri = []

    # ri.append([cvt2(x0, Geometry.TYPE.ZONOTOPE)])

    @jit(nopython=True, parallel=True)
    def parallel_reach(bounds):
        for this_bound in bounds:
            lin_sys = LinearSystemSimple(xa, ub)
            opts = ReachLinearZonoAlgo1.Settings()
            opts.t_end = 0.5
            opts.step = 0.01
            opts.eta = 4

            opts.x0 = this_bound
            _, this_ri, _, _ = ReachLinearZonoAlgo1.reach(lin_sys, opts)
            ri.append(this_ri)

    parallel_reach(x0_bounds)

    plot(ri, [0, 1])

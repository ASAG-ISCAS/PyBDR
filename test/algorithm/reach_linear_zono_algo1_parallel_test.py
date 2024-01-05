import numpy as np
from pybdr.geometry import Zonotope, Geometry, Interval
from pybdr.geometry.operation import cvt2
from pybdr.dynamic_system import LinearSystemSimple
from pybdr.algorithm import ReachLinearZonoAlgo1
from pybdr.util.visualization import plot
from concurrent.futures import ProcessPoolExecutor, as_completed
from pybdr.geometry.operation import boundary

"""
Reachable set for linear time-invariant systems without input
"""

xa = [[-1, -4], [4, -1]]
ub = np.array([[1], [1]])

lin_sys = LinearSystemSimple(xa, ub)
opts = ReachLinearZonoAlgo1.Settings()
opts.t_end = 5
opts.step = 0.04
opts.eta = 4


def parallel_reach(x):
    return ReachLinearZonoAlgo1.reach(lin_sys, opts, x)


if __name__ == '__main__':

    x0 = Interval([0.9, 0.9], [1.1, 1.1])
    x0_bounds = boundary(x0, 0.0001, Geometry.TYPE.ZONOTOPE)

    print(len(x0_bounds))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(parallel_reach, bound) for bound in x0_bounds]

        for future in as_completed(futures):
            try:
                _, ri, _, _ = future.result()
            except Exception as exc:
                print(exc)

    plot(ri, [0, 1])

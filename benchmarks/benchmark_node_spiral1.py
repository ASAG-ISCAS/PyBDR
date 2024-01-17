import numpy as np

from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.model import *
from pybdr.util.visualization import plot
from pybdr.geometry.operation import boundary

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 1.0
    options.step = 0.02
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.0, -2.0], [4.0, 2.0])

    cell_nums = {16: 1.1, 20: 1.0, 24: 0.7, 28: 0.6, 32: 0.55}

    xs = boundary(z, cell_nums[20], Geometry.TYPE.ZONOTOPE)

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(neural_ode_spiral1, [2, 1], options, xs)

    # visualize the results
    plot(ri_with_bound, [0, 1])

import numpy as np
from pybdr.algorithm import ASB2008CDC, ASB2008CDC
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary
from pybdr.model import *
from pybdr.util.visualization import plot

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 5.0
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0.8, 1.8], [2.0, 3.0])

    cell_nums_s = {12: 0.3, 16: 0.25, 20: 0.18}
    cell_nums_m = {8: 0.6, 12: 0.4, 16: 0.3, 20: 0.22, 24: 0.18, 28: 0.16, 32: 0.14, 36: 0.12}
    cell_nums_l = {20: 0.3, 24: 0.2, 28: 0.18, 32: 0.16, 36: 0.15}

    xs = boundary(z, cell_nums_l[36], Geometry.TYPE.ZONOTOPE)

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(vanderpol, [2, 1], options, xs)

    # visualize the results
    plot(ri_with_bound, [0, 1])

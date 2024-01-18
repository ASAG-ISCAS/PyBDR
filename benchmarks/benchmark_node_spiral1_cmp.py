import numpy as np

from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.model import *
from pybdr.util.visualization import plot, plot_cmp
from pybdr.geometry.operation import boundary, cvt2
from pybdr.util.functional import performance_counter_start, performance_counter

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 1
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 2

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0, -0.5], [1, 0.5])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 2, Geometry.TYPE.ZONOTOPE)

    print(len(xs))

    this_time = performance_counter_start()
    ri_without_bound, rp_without_bound = ASB2008CDC.reach(neural_ode_spiral1, [2, 1], options, x0)
    this_time = performance_counter(this_time, "reach_without_bound")

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(neural_ode_spiral1, [2, 1], options, xs)
    this_time = performance_counter(this_time, "reach_with_bound")

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"])

import numpy as np
from pybdr.algorithm import ASB2008CDC, ASB2008CDC
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary, cvt2
from pybdr.model import *
from pybdr.util.visualization import plot

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([1.23, 2.34], [1.57, 2.46])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    this_time = performance_counter_start()
    ri_without_bound, rp_without_bound = ASB2008CDC.reach(vanderpol, [2, 1], options, x0)
    this_time = performance_counter(this_time, 'reach_without_bound')

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(vanderpol, [2, 1], options, xs)
    this_time = performance_counter(this_time, 'reach_with_bound')

    # visualize the results
    plot(ri_without_bound, [0, 1])
    plot(ri_with_bound, [0, 1])

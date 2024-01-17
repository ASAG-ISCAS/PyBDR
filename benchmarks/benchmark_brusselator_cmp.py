import numpy as np
from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary, cvt2
from pybdr.model import *
from pybdr.util.visualization import plot_cmp

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.0
    options.step = 0.02
    options.tensor_order = 2
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval.identity(2) * 0.1 + 0.2
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 0.3, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(brusselator, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(brusselator, [2, 1], options, xs)

    # visualize the results
    xlim, ylim = [0, 2], [0, 2.4]
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"])

import numpy as np

from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary, cvt2
from pybdr.model import *
from pybdr.util.visualization import plot_cmp

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 3
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([-0.2, 2.8], [0.2, 3.2])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(synchronous_machine, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(synchronous_machine, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"])

# if __name__ == '__main__':
#     # init dynamic system
#     system = NonLinSys(Model(synchronous_machine, [2, 1]))
#
#     # settings for the computation
#     options = ASB2008CDC.Options()
#     options.t_end = 3
#     options.step = 0.01
#     options.tensor_order = 3
#     options.taylor_terms = 4
#
#     options.u = Zonotope.zero(1, 1)
#     options.u_trans = np.zeros(1)
#
#     # settings for the using geometry
#     Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
#     Zonotope.ORDER = 50
#
#     # z = Zonotope([0, 3], np.diag([0.2, 0.2]))
#     z = Interval([-0.2, 2.8], [0.2, 3.2])
#
#     # reachable sets computation without boundary analysis
#     options.r0 = [cvt2(z, Geometry.TYPE.ZONOTOPE)]
#     ti_whole, tp_whole, _, _ = ASB2008CDC.reach(system, options)
#
#     # reachable sets computation with boundary analysis
#     options.r0 = boundary(z, 1, Geometry.TYPE.ZONOTOPE)
#     ti_bound, tp_bound, _, _ = ASB2008CDC.reach(system, options)
#
#     # visualize the results
#     plot_cmp([tp_whole, tp_bound], [0, 1], cs=["#FF5722", "#303F9F"])

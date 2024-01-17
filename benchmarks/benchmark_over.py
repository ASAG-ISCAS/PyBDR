from __future__ import annotations
import numpy as np

from pybdr.geometry import Geometry, Zonotope, Interval
from pybdr.model import vanderpol
from pybdr.algorithm import ASB2008CDC
from pybdr.geometry.operation import boundary, cvt2
from pybdr.util.visualization import plot_cmp

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.74
    options.step = 0.01
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

    ri_without_bound, rp_without_bound = ASB2008CDC.reach(vanderpol, [2, 1], options, x0)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(vanderpol, [2, 1], options, xs)

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"])

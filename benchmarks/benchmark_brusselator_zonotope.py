import numpy as np
import codac
from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Polytope, Geometry
from pybdr.geometry.operation import cvt2
from pybdr.model import *
from pybdr.util.visualization import plot_cmp, plot
from pybdr.util.functional import extract_boundary
from pybdr.util.functional import performance_counter, performance_counter_start

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 10.0
    options.step = 0.05
    options.tensor_order = 2
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    data = np.array(
    [[0.0, 0.24, -0.10, 0.03, -0.08],  # x
            [0.0, 0.05, 0.18, -0.12, -0.04]],
        dtype=float,
    )
    init_set = Zonotope(data[:, 0], data[:, 1:])
    init_box = codac.IntervalVector([[-0.5, 0.5], [-0.5, 0.5]])

    this_time = performance_counter_start()
    xs = extract_boundary(init_box, init_set, eps=0.04)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(brusselator, [2, 1], options, xs)
    this_time = performance_counter(this_time, 'reach_without_bound')

    # visualize the results
    plot(ri_with_bound, [0, 1], c="#303F9F")

    boundary_boxes = []
    for i in xs:
        boundary_boxes.append(cvt2(i, Geometry.TYPE.INTERVAL))
    plot(boundary_boxes, [0, 1], init_set=init_set)

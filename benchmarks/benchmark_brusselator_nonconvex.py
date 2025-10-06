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

    x = codac.VectorVar(2)
    f = codac.AnalyticFunction([x], [(0.31) * 1 + (2.57) * x[0] + (-0.00) * x[1] + (-4.58) * x[0] ** 2 + (-2.71) * x[0] * x[1] + (13.63) * x[1] ** 2 + (-9.76) * x[0] ** 3 + (-1.94) * x[0] ** 2 * x[1] + (-6.82) * x[0] * x[1] ** 2 + (0.75) * x[1] ** 3 + (-16.03) * x[0] ** 4 + (7.56) * x[0] ** 3 * x[1] + (15.39) * x[0] ** 2 * x[1] ** 2 + (6.70) * x[0] * x[1] ** 3 + (-83.93) * x[1] ** 4])
    init_box = Interval([-0.5, -0.5], [0.5, 0.5])

    this_time = performance_counter_start()
    xs = extract_boundary(init_interval=init_box, init_set=f, eps=0.04)
    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(brusselator, [2, 1], options, xs)
    this_time = performance_counter(this_time, 'reach_without_bound')

    # visualize the results
    plot(ri_with_bound, [0, 1], c="#303F9F")

    boundary_boxes = []
    for i in xs:
        boundary_boxes.append(cvt2(i, Geometry.TYPE.INTERVAL))
    constraint = ("(0.31)*1 + (2.57)*x + (-0.00)*y + (-4.58)*x^2 + (-2.71)*x y + (13.63)*y^2 + (-9.76)*x^3 + "
                  "(-1.94)*x^2 y + (-6.82)*x y^2 + (0.75)*y^3 + (-16.03)*x^4 + (7.56)*x^3 y + (15.39)*x^2 y^2 + "
                  "(6.70)*x y^3 + (-83.93)*y^4")
    constraint = constraint.replace(" y", "*y")
    plot(boundary_boxes, [0, 1], xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], init_set=constraint)


import numpy as np
from pybdr.geometry import Interval, Zonotope, Geometry
from pybdr.geometry.operation import cvt2, partition, boundary
from pybdr.algorithm import ASB2008CDC
from pybdr.dynamic_system import NonLinSys
from pybdr.model import *
from pybdr.util.visualization import plot, plot_cmp
from pybdr.util.functional import performance_counter, performance_counter_start

if __name__ == '__main__':
    time_start = performance_counter_start()
    # init dynamic system
    system = NonLinSys(Model(lotka_volterra_2d, [2, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 2.2
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval.identity(2) * 0.5 + 3

    options.r0 = [cvt2(z, Geometry.TYPE.ZONOTOPE)]
    _, tp_whole, _, _ = ASB2008CDC.reach(system, options)

    # --------------------------------------------------------
    # options.r0 = partition(z, 1, Geometry.TYPE.ZONOTOPE)
    # 4
    # ASB2008CDC cost: 43.344963666000005s
    # --------------------------------------------------------
    # options.r0 = partition(z, 0.5, Geometry.TYPE.ZONOTOPE)
    # 9
    # ASB2008CDC cost: 3868912500001s
    # --------------------------------------------------------
    options.r0 = partition(z, 0.2, Geometry.TYPE.ZONOTOPE)
    # 36
    # ASB2008CDC cost: 317.59988937500003s
    # --------------------------------------------------------
    # options.r0 = partition(z, 0.1, Geometry.TYPE.ZONOTOPE)

    print(len(options.r0))

    _, tp_part_00, _, _ = ASB2008CDC.reach(system, options)
    performance_counter(time_start, "ASB2008CDC")

    plot_cmp([tp_whole, tp_part_00], [0, 1], cs=["#FF5722", "#303F9F"])

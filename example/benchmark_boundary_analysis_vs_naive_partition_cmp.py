import numpy as np
from pybdr.geometry import Interval, Zonotope, Geometry
from pybdr.geometry.operation import cvt2, partition, boundary
from pybdr.algorithm import ASB2008CDCParallel
from pybdr.dynamic_system import NonLinSys
from pybdr.model import *
from pybdr.util.visualization import plot, plot_cmp
from pybdr.util.functional import performance_counter, performance_counter_start

if __name__ == '__main__':
    # tuples_list = [(1, 'a', 'x', 'u'), (2, 'b', 'y', 'v'), (3, 'c', 'z', 'w')]
    #
    # lists_list = [list(t) for t in zip(*tuples_list)]
    #
    # print(lists_list)
    #
    # exit(False)
    time_start = performance_counter_start()
    # init dynamic system
    # system = NonLinSys(Model(lotka_volterra_2d, [2, 1]))

    # settings for the computation
    options = ASB2008CDCParallel.Options()
    options.t_end = 2.2
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    init_set = Interval.identity(2) * 0.5 + 3

    _, tp_whole = ASB2008CDCParallel.reach(lotka_volterra_2d, [2, 1], options,
                                           cvt2(init_set, Geometry.TYPE.ZONOTOPE))

    # NAIVE PARTITION
    # --------------------------------------------------------
    # options.r0 = partition(z, 1, Geometry.TYPE.ZONOTOPE)
    # xs = partition(init_set, 1, Geometry.TYPE.ZONOTOPE)
    # 4
    # ASB2008CDCParallel.reach_parallel cost: 20.126129667s
    # --------------------------------------------------------
    # xs = partition(init_set, 0.5, Geometry.TYPE.ZONOTOPE)
    # 9
    # ASB2008CDCParallel.reach_parallel cost: 23.938516459000002s
    # --------------------------------------------------------
    xs = partition(init_set, 0.2, Geometry.TYPE.ZONOTOPE)
    # 36
    # ASB2008CDCParallel.reach_parallel cost: 65.447113125s
    # --------------------------------------------------------

    #  BOUNDAYR ANALYSIS
    # --------------------------------------------------------
    xs = boundary(init_set, 1, Geometry.TYPE.ZONOTOPE)
    # 8
    # ASB2008CDCParallel.reach_parallel cost: 22.185758250000003s

    print(len(xs))

    _, tp_part_00 = ASB2008CDCParallel.reach_parallel(lotka_volterra_2d, [2, 1], options, xs)

    performance_counter(time_start, "ASB2008CDCParallel.reach_parallel")

    plot_cmp([tp_whole, tp_part_00], [0, 1], cs=["#FF5722", "#303F9F"])

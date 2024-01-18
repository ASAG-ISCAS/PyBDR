import numpy as np
from pybdr.geometry import Interval, Zonotope, Geometry
from pybdr.geometry.operation import cvt2, partition, boundary
from pybdr.algorithm import ASB2008CDC
from pybdr.model import *
from pybdr.util.visualization import plot_cmp
from pybdr.util.functional import performance_counter, performance_counter_start

if __name__ == '__main__':
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

    init_set = Interval.identity(2) * 0.5 + 3

    this_time = performance_counter_start()
    _, rp_without_bound = ASB2008CDC.reach(lotka_volterra_2d, [2, 1], options, cvt2(init_set, Geometry.TYPE.ZONOTOPE))
    this_time = performance_counter(this_time, 'reach_without_bound')

    # NAIVE PARTITION
    # --------------------------------------------------------
    # options.r0 = partition(z, 1, Geometry.TYPE.ZONOTOPE)
    xs_naive_partition_4 = partition(init_set, 1, Geometry.TYPE.ZONOTOPE)
    # 4
    # ASB2008CDCParallel.reach_parallel cost: 20.126129667s MacOS
    # --------------------------------------------------------
    xs_naive_partition_9 = partition(init_set, 0.5, Geometry.TYPE.ZONOTOPE)
    # 9
    # ASB2008CDCParallel.reach_parallel cost: 23.938516459000002s MacOS
    # --------------------------------------------------------
    xs_naive_partition_36 = partition(init_set, 0.2, Geometry.TYPE.ZONOTOPE)
    # 36
    # ASB2008CDCParallel.reach_parallel cost: 65.447113125s MacOS
    # --------------------------------------------------------

    #  BOUNDAYR ANALYSIS
    # --------------------------------------------------------
    xs_with_bound_8 = boundary(init_set, 1, Geometry.TYPE.ZONOTOPE)
    # 8
    # ASB2008CDCParallel.reach_parallel cost: 22.185758250000003s MacOS

    print(len(xs_naive_partition_4))
    print(len(xs_naive_partition_9))
    print(len(xs_naive_partition_36))
    print(len(xs_with_bound_8))

    this_time = performance_counter(this_time, 'partition and boundary')

    _, rp_naive_partition_4 = ASB2008CDC.reach_parallel(lotka_volterra_2d, [2, 1], options, xs_naive_partition_4)
    this_time = performance_counter(this_time, 'reach naive partition 4')

    _, rp_naive_partition_9 = ASB2008CDC.reach_parallel(lotka_volterra_2d, [2, 1], options, xs_naive_partition_9)
    this_time = performance_counter(this_time, 'reach naive partition 9')

    _, rp_naive_partition_36 = ASB2008CDC.reach_parallel(lotka_volterra_2d, [2, 1], options, xs_naive_partition_36)
    this_time = performance_counter(this_time, 'reach naive partition 36')

    _, rp_with_bound_8 = ASB2008CDC.reach_parallel(lotka_volterra_2d, [2, 1], options, xs_with_bound_8)
    this_time = performance_counter(this_time, 'reach with bound 8')

    plot_cmp([rp_without_bound, rp_naive_partition_4], [0, 1], cs=["#FF5722", "#303F9F"])
    plot_cmp([rp_without_bound, rp_naive_partition_9], [0, 1], cs=["#FF5722", "#303F9F"])
    plot_cmp([rp_without_bound, rp_naive_partition_36], [0, 1], cs=["#FF5722", "#303F9F"])
    plot_cmp([rp_without_bound, rp_with_bound_8], [0, 1], cs=["#FF5722", "#303F9F"])

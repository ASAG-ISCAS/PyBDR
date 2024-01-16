import numpy as np
from pybdr.algorithm import ASB2008CDC,ASB2008CDCParallel
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.dynamic_system import NonLinSys
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary
from pybdr.model import *
from pybdr.util.visualization import plot
from save_results import save_result
from pybdr.geometry.operation.convert import cvt2

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDCParallel.Options()
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    options.t_end = 5.0
    # options.t_end = 7.0
    # options.t_end = 9.0
    # Z = Interval([1.0, 2.0], [1.8, 2.8])
    # Z = Interval([0.9, 1.9], [1.9, 2.9])
    Z = Interval([0.8, 1.8], [2.0, 3.0])

    cell_nums_s = {12: 0.3, 16: 0.25, 20: 0.18}
    cell_nums_m = {8: 0.6, 12: 0.4, 16: 0.3, 20: 0.22, 24: 0.18, 28: 0.16, 32: 0.14, 36: 0.12}
    cell_nums_l = {20: 0.3, 24: 0.2, 28: 0.18, 32: 0.16, 36: 0.15}
    # Reachable sets computed with boundary analysis
    options.r0 = boundary(Z, cell_nums_l[36], Geometry.TYPE.ZONOTOPE)
    # Reachable sets computed without boundary analysis
    # options.r0 = [cvt2(Z, Geometry.TYPE.ZONOTOPE)]
    print("The number of divided sets: ", len(options.r0))

    time_tag = performance_counter_start()
    ri, rp = ASB2008CDCParallel.reach_parallel(vanderpol, [2, 1], options, options.r0)
    performance_counter(time_tag, "ASB2008CDCParallel")

    # save the results
    save_result(ri, "vanderpol_" + str(len(options.r0)))

    # visualize the results
    plot(ri, [0, 1])

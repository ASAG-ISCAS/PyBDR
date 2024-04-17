import numpy as np
from pybdr.algorithm import ASB2008CDC,ASB2008CDCParallel
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary
from pybdr.model import *
from pybdr.util.visualization import plot
from pybdr.geometry.operation.convert import cvt2
from save_results import save_result

if __name__ == '__main__':

    options = ASB2008CDCParallel.Options()
    options.t_end = 2.2
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)
    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    Z = Interval([2.5, 2.5], [3.5, 3.5])
    # Reachable sets computed with boundary analysis
    # options.r0 = boundary(Z, 0.6, Geometry.TYPE.ZONOTOPE)

    num = 6
    r0 = []
    for i in range(num):
        for j in range(num):
            Z_tmp = Interval([2.5 + 1 / num * i, 2.5 + 1 / num * j], [2.5 + 1 / num * (i+1), 2.5 + 1 / num * (j+1)])
            r0.append(cvt2(Z_tmp, Geometry.TYPE.ZONOTOPE))
    options.r0 = r0
    # Reachable sets computed without boundary analysis
    # options.r0 = [cvt2(Z, Geometry.TYPE.ZONOTOPE)]
    print("The number of divided sets: ", len(options.r0))

    time_tag = performance_counter_start()
    ri, rp = ASB2008CDCParallel.reach_parallel(lotka_volterra_2d, [2, 1], options, options.r0)
    performance_counter(time_tag, "ASB2008CDCParallel")

    # save the results
    # save_result(ri, "synchronous_" + str(len(options.r0)))

    # visualize the results
    plot(ri, [0, 1])
 
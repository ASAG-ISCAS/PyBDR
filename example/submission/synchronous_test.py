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
    options.t_end = 7.0
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)
    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    Z = Interval([-0.7, 2.3], [0.7, 3.7])
    cell_nums = {12: 0.5, 16: 0.4, 20: 0.3, 24: 0.25}
    # Reachable sets computed with boundary analysis
    options.r0 = boundary(Z, cell_nums[24], Geometry.TYPE.ZONOTOPE)
    # Reachable sets computed without boundary analysis
    # options.r0 = [cvt2(Z, Geometry.TYPE.ZONOTOPE)]
    print("The number of divided sets: ", len(options.r0))

    time_tag = performance_counter_start()
    ri, rp = ASB2008CDCParallel.reach_parallel(synchronous_machine, [2, 1], options, options.r0)
    performance_counter(time_tag, "ASB2008CDCParallel")

    # save the results
    save_result(ri, "synchronous_" + str(len(options.r0)))

    # visualize the results
    plot(ri, [0, 1])

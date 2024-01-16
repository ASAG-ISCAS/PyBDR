import numpy as np

from pybdr.algorithm import ASB2008CDC,ASB2008CDCParallel
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary
from pybdr.model import *
from pybdr.util.visualization import plot
from save_results import save_result
from pybdr.geometry.operation.convert import cvt2

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDCParallel.Options()
    options.t_end = 5.0
    options.step = 0.01
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    Z = Interval([-0.1, 3.9], [0.1, 4.1])
    cell_nums = {12: 0.05, 16: 0.06, 20: 0.045}
    # Reachable sets computed with boundary analysis
    options.r0 = boundary(Z, cell_nums[20], Geometry.TYPE.ZONOTOPE)
    # Reachable sets computed without boundary analysis
    # options.r0 = [cvt2(Z, Geometry.TYPE.ZONOTOPE)]
    print("The number of divided sets: ", len(options.r0))


    time_tag = performance_counter_start()
    ri, rp = ASB2008CDCParallel.reach_parallel(brusselator, [2, 1], options, options.r0)
    performance_counter(time_tag, "ASB2008CDCParallel")

    # save the results
    save_result(ri, "brusselator_" + str(len(options.r0)))

    # visualize the results
    # plot(ri, [0, 1])

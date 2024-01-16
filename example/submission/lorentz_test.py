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

    options = ASB2008CDCParallel.Options()
    options.t_end = 1.0
    options.step = 0.02
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    Z = Interval([-11, -3, -3], [3, 11, 11])
    cell_nums = {216: 2.5, 294: 2.3, 384: 2}
    # Reachable sets computed with boundary analysis
    options.r0 = boundary(Z, cell_nums[384], Geometry.TYPE.ZONOTOPE)
    # Reachable sets computed without boundary analysis
    # options.r0 = [cvt2(Z, Geometry.TYPE.ZONOTOPE)]
    print("The number of divided sets: ", len(options.r0))

    time_tag = performance_counter_start()
    ri, rp = ASB2008CDCParallel.reach_parallel(lorentz, [3, 1], options, options.r0)
    performance_counter(time_tag, "ASB2008CDCParallel")

    # save the results
    save_result(ri, "lorentz_" + str(len(options.r0)))

    # visualize the results
    plot(ri, [0, 1])
    plot(ri, [1, 2])
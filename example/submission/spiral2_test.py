import numpy as np
np.seterr(divide='ignore', invalid='ignore')
# sys.path.append("./../../") uncomment this line if you need to add path manually
from pybdr.algorithm import ASB2008CDC,ASB2008CDCParallel
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.model import *
from pybdr.util.visualization import plot
from pybdr.geometry.operation import boundary
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.geometry.operation.convert import cvt2
from save_results import save_result

if __name__ == '__main__':
    # settings for the computation
    options = ASB2008CDCParallel.Options()
    options.t_end = 7.0
    options.step = 0.1
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)
    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    Z = Interval([-4.0, -4.0], [-2.0, -2.0])
    cell_nums = {16: 1.1, 20: 1.0, 24: 0.7, 28: 0.65, 32: 0.52}
    # Reachable sets computed with boundary analysis
    options.r0 = boundary(Z, cell_nums[32], Geometry.TYPE.ZONOTOPE)
    # Reachable sets computed without boundary analysis
    # options.r0 = [cvt2(Z, Geometry.TYPE.ZONOTOPE)]
    print("The number of divided sets: ", len(options.r0))

    # reachable sets computation
    time_tag = performance_counter_start()
    ri, rp = ASB2008CDCParallel.reach_parallel(spiral2, [2,1], options, options.r0)
    performance_counter(time_tag, "ASB2008CDCParallel")

    # save the results
    save_result(ri, "spiral2_"+str(len(options.r0)))

    # visualize the results
    plot(ri, [0, 1])
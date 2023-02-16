import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import sys
sys.path.append("./../../")
from pyrat.algorithm import ALTHOFF2013HSCC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry, Zonotope
from pyrat.geometry.operation import cvt2
from pyrat.model import *
from pyrat.util.visualization import plot



# init dynamic system
system = NonLinSys(Model(vanderpol, [2, 1]))

# settings for the computation
options = ALTHOFF2013HSCC.Options()
options.t_end = 6.74
options.step = 0.005
options.taylor_terms = 4
options.tensor_order = 3
options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
options.u = Zonotope.zero(1, 1)
options.u_trans = np.zeros(1)

# settings for using Zonotope
Zonotope.ORDER = 50
Zonotope.INTERMEDIATE_ORDER = 50
Zonotope.ERROR_ORDER = 20

# reachable sets
ti, tp, _, _ = ALTHOFF2013HSCC.reach(system, options)

# visualize the results
plot(tp, [0, 1])
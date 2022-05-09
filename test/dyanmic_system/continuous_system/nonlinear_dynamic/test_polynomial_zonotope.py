import numpy as np
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.model import *
from pyrat.util.visualization import vis2d


def test_vanDerPol():
    system = NonLinSys.Entity(VanDerPol())

    # setting for the computation
    option = NonLinSys.Option.Polynomial()
    option.t_end = 6.74
    option.steps = 1348
    option.taylor_terms = 4
    option.tensor_order = 3
    option.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    option.u = Zonotope.zero(1, 1)
    option.u_trans = np.zeros(1)

    # setting for using Zonotope
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    results = system.reach(option)
    vis2d(results, [0, 1])

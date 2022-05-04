import inspect

import numpy as np
from pyrat.dynamic_system import NonLinSys
from pyrat.model import *
from pyrat.geometry import Zonotope
from pyrat.misc import Reachable, Set
from pyrat.util.visualization import vis2d


def test_vanDerPol():
    system = NonLinSys.Entity(VanDerPol())

    # setting for the computation
    option = NonLinSys.Option.Linear()
    option.t_end = 4
    option.steps = 1000
    option.tensor_order = 2
    option.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    option.u = Zonotope([0], [[0.005]])
    option.u = Zonotope.zero(1, 1)
    option.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [0, 1])

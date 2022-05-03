import inspect

import numpy as np
from pyrat.dynamic_system import NonLinSys
from pyrat.model import *
from pyrat.geometry import Zonotope
from pyrat.misc import Reachable, Set
from pyrat.util.visualization import vis2d


def test_tank6Eq():
    system = NonLinSys.Entity(Tank6Eq())

    # setting for the computation
    option = NonLinSys.Option.Linear()
    option.t_end = 800
    option.steps = 150
    option.tensor_order = 2
    option.r0 = [Zonotope([2, 4, 4, 2, 10, 4], np.eye(6) * 0)]
    option.u = Zonotope([0], [[0.005]])
    option.u_trans = np.zeros(1)
    option.taylor_terms = 4

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [0, 1])
    vis2d(results, [2, 3])
    vis2d(results, [4, 5])

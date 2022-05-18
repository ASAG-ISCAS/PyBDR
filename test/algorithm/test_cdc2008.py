import numpy as np
from pyrat.algorithm import CDC2008
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.model import *
from pyrat.util.visualization import vis2d


def test_van_der_pol():
    # init dynamic system
    system = NonLinSys.Entity(VanDerPol())

    # settings for the computation
    options = CDC2008.Options()
    options.t_end = 4
    options.step = 0.004
    options.tensor_order = 2
    options.taylor_terms = 4
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    results = CDC2008.reach(system, options)

    # visualize the results
    vis2d(results, [0, 1])


def test_tank6eq():
    # init dynamic system
    system = NonLinSys.Entity(Tank6Eq())

    # settings for the computation
    options = CDC2008.Options()
    options.t_end = 400
    options.step = 4
    options.tensor_order = 2
    options.taylor_terms = 4
    options.r0 = [Zonotope([2, 4, 4, 2, 10, 4], np.eye(6) * 0.2)]
    options.u = Zonotope([0], [[0.005]])
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    results = CDC2008.reach(system, options)

    # visualize the results
    vis2d(results, [0, 1])
    vis2d(results, [2, 3])
    vis2d(results, [4, 5])


def test_laub_loomis():
    # init dynamic system
    system = NonLinSys.Entity(LaubLoomis())

    # settings for the computation
    options = CDC2008.Options()
    options.t_end = 20
    options.step = 0.04
    options.tensor_order = 2
    options.taylor_terms = 4
    options.r0 = [Zonotope([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45], np.eye(7) * 0)]
    options.u = Zonotope([0], [[0.005]])
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = CDC2008.reach(system, options)
    vis2d(results, [0, 1])
    vis2d(results, [2, 4])
    vis2d(results, [5, 6])
    vis2d(results, [1, 5])
    vis2d(results, [4, 6])

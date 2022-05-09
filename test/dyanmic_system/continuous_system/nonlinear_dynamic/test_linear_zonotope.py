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
    option.taylor_terms = 4
    option.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    option.u = Zonotope([0], [[0.005]])
    option.u = Zonotope.zero(1, 1)
    option.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [0, 1])


def test_tank6Eq():
    system = NonLinSys.Entity(Tank6Eq())
    # setting for the computation
    option = NonLinSys.Option.Linear()
    option.t_end = 400
    option.steps = 100
    option.tensor_order = 2
    option.taylor_terms = 4
    option.r0 = [Zonotope([2, 4, 4, 2, 10, 4], np.eye(6) * 0.2)]
    option.u = Zonotope([0], [[0.005]])
    option.u = Zonotope.zero(1, 1)
    option.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [0, 1])
    vis2d(results, [2, 3])
    vis2d(results, [4, 5])


def test_laub_loomis():
    system = NonLinSys.Entity(LaubLoomis())
    # setting for the computation
    option = NonLinSys.Option.Linear()
    option.t_end = 20
    option.steps = 500
    option.tensor_order = 2
    option.taylor_terms = 4
    option.r0 = [Zonotope([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45], np.eye(7) * 0)]
    option.u = Zonotope([0], [[0.005]])
    option.u = Zonotope.zero(1, 1)
    option.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [0, 1])
    vis2d(results, [2, 4])
    vis2d(results, [5, 6])
    vis2d(results, [1, 5])
    vis2d(results, [4, 6])


def test_ltv():
    system = NonLinSys.Entity(LTV())
    # setting for the computation
    option = NonLinSys.Option.Linear()
    option.t_end = 5
    option.steps = 500
    option.tensor_order = 2
    option.taylor_terms = 4
    option.r0 = [Zonotope([1.25, 5.25, 0], np.eye(3) * 0)]
    option.u = Zonotope([0], [[0.005]])
    option.u = Zonotope.zero(4, 4)
    option.u_trans = np.zeros(4)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [0, 1])


def test_genetic():
    system = NonLinSys.Entity(GeneticModel())
    # setting for the computation
    option = NonLinSys.Option.Linear()
    option.t_end = 3
    option.steps = 500
    option.tensor_order = 2
    option.taylor_terms = 4
    option.r0 = [Zonotope([1.0, 1.3, 0.1, 0.1, 0.1, 1.3, 2.5, 0.6, 1.3], np.eye(9) * 0)]
    option.u = Zonotope([0], [[0.005]])
    option.u = Zonotope.zero(1)
    option.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [2, 4])
    vis2d(results, [3, 5])


def test_p53_small():
    system = NonLinSys.Entity(P53Small())
    # setting for the computation
    option = NonLinSys.Option.Linear()
    option.t_end = 10
    option.steps = 500
    option.tensor_order = 2
    option.taylor_terms = 4
    option.r0 = [Zonotope([20.0] * 6, np.eye(6) * 0)]
    option.u = Zonotope([0], [[0.005]])
    option.u = Zonotope.zero(1)
    option.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    results = system.reach(option)
    vis2d(results, [0, 1])
    vis2d(results, [2, 5])

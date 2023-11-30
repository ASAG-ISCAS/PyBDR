import numpy as np
from pybdr.dynamic_system import NonLinSysOld
from pybdr.geometry import Zonotope
from pybdr.model import *
from pybdr.util.visualization import vis2d


def test_vanDerPol():
    system = NonLinSysOld.Entity(VanDerPol())

    # setting for the computation
    option = NonLinSysOld.Option.Polynomial()
    option.t_end = 6.74
    option.steps = 1348
    option.taylor_terms = 8
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


def test_tank6Eq():
    system = NonLinSysOld.Entity(Tank6Eq())
    # setting for the computation
    option = NonLinSysOld.Option.Polynomial()
    option.t_end = 400
    option.steps = 100
    option.tensor_order = 3
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
    system = NonLinSysOld.Entity(LaubLoomis())
    # setting for the computation
    option = NonLinSysOld.Option.Polynomial()
    option.t_end = 20
    option.steps = 500
    option.tensor_order = 3
    option.taylor_terms = 4
    option.r0 = [Zonotope([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45], np.eye(7) * 0.01)]
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
    system = NonLinSysOld.Entity(LTV())
    # setting for the computation
    option = NonLinSysOld.Option.Polynomial()
    option.t_end = 5
    option.steps = 500
    option.tensor_order = 3
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
    system = NonLinSysOld.Entity(GeneticModel())
    # setting for the computation
    option = NonLinSysOld.Option.Polynomial()
    option.t_end = 3
    option.steps = 500
    option.tensor_order = 3
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
    system = NonLinSysOld.Entity(P53Small())
    # setting for the computation
    option = NonLinSysOld.Option.Polynomial()
    option.t_end = 7
    option.steps = 300
    option.tensor_order = 3
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

import numpy as np

from pyrat.algorithm import ASB2008CDC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.model import *
from pyrat.util.visualization import plot


def test_van_der_pol():
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 3.5
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 4
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    ti, tp, _, _ = ASB2008CDC.reach(system, options)

    tp = [[r.geometry for r in l] for l in tp]
    # visualize the results
    plot(tp, [0, 1])


def test_tank6eq():
    # init dynamic system
    system = NonLinSys(Model(tank6eq, [6, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
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
    ti, tp, _, _ = ASB2008CDC.reach(system, options)

    tp = [[r.geometry for r in l] for l in tp]

    # visualize the results
    plot(tp, [0, 1])
    plot(tp, [2, 3])
    plot(tp, [4, 5])


def test_laub_loomis():
    # init dynamic system
    system = NonLinSys(Model(laubloomis, [7, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 20
    options.step = 0.04
    options.tensor_order = 2
    options.taylor_terms = 4
    options.r0 = [Zonotope([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45], np.eye(7) * 0.01)]
    options.u = Zonotope([0], [[0.005]])
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50

    ti, tp, _, _ = ASB2008CDC.reach(system, options)

    tp = [[r.geometry for r in l] for l in tp]

    plot(tp, [0, 1])
    plot(tp, [2, 4])
    plot(tp, [5, 6])
    plot(tp, [1, 5])
    plot(tp, [4, 6])

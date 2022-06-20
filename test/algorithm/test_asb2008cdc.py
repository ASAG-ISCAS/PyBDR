import numpy as np
from pyrat.algorithm import ASB2008CDC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.model import *
from pyrat.util.visualization import plot


def test_van_der_pol():
    # init dynamic system
    system = NonLinSys.Entity(VanDerPol())

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 3.5
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
    results = ASB2008CDC.reach(system, options)

    geos = []
    for tps in results.tps:
        for tp in tps:
            geos.append(tp.geometry)

    # visualize the results
    plot(geos, [0, 1])


def test_tank6eq():
    # init dynamic system
    system = NonLinSys.Entity(Tank6Eq())

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
    results = ASB2008CDC.reach(system, options)

    geos = []
    for tps in results.tps:
        for tp in tps:
            geos.append(tp.geometry)

    # visualize the results
    plot(geos, [0, 1])
    plot(geos, [2, 3])
    plot(geos, [4, 5])


def test_laub_loomis():
    # init dynamic system
    system = NonLinSys.Entity(LaubLoomis())

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

    results = ASB2008CDC.reach(system, options)

    geos = []
    for tps in results.tps:
        for tp in tps:
            geos.append(tp.geometry)

    plot(geos, [0, 1])
    plot(geos, [2, 4])
    plot(geos, [5, 6])
    plot(geos, [1, 5])
    plot(geos, [4, 6])

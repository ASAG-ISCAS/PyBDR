import numpy as np
from pyrat.algorithm import HSCC2013
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry, Zonotope
from pyrat.model import *
from pyrat.util.visualization import plot


def test_van_der_pol_using_zonotope():
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = HSCC2013.Options()
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
    ti, tp, _, _ = HSCC2013.reach(system, options)

    # tp = [[r.geometry for r in l] for l in tp]

    # visualize the results
    plot(tp, [0, 1])


def test_van_der_pol_using_polyzonotope():
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = HSCC2013.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.taylor_terms = 4
    options.tensor_order = 3
    poly_zono = cvt2(
        Zonotope([1.4, 2.4], np.diag([0.17, 0.06])), Geometry.TYPE.POLY_ZONOTOPE
    )
    options.r0 = [poly_zono]
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for poly_zonotope
    # PolyZonotope.MAX_DEPTH_GEN_ORDER = 50
    # PolyZonotope.MAX_POLY_ZONO_RATIO = 0.01
    # PolyZonotope.RESTRUCTURE_TECH = PolyZonotope.METHOD.RESTRUCTURE.REDUCE_PCA

    # reachable sets
    results = HSCC2013.reach(system, options)

    geos = []
    for tps in results.tps:
        for tp in tps:
            geos.append(tp.geometry)

    # visualize the results
    plot(geos, [0, 1])


def test_tank6eq():
    # init dynamic system
    system = NonLinSys(Model(tank6eq, [6, 1]))

    # settings for the computations
    options = HSCC2013.Options()
    options.t_end = 400
    options.step = 4
    options.tensor_order = 3
    options.taylor_terms = 4
    options.r0 = [Zonotope([2, 4, 4, 2, 10, 4], np.eye(6) * 0.2)]
    options.u = Zonotope([0], [[0.005]])
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for using geometry
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD
    Zonotope.ORDER = 50
    Zonotope.INTERMEDIATE_ORDER = 50
    Zonotope.ERROR_ORDER = 20

    results = HSCC2013.reach(system, options)

    geos = []
    for tps in results.tps:
        for tp in tps:
            geos.append(tp.geometry)

    plot(geos, [0, 1])
    plot(geos, [2, 3])
    plot(geos, [4, 5])

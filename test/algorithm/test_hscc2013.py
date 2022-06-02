import numpy as np

from pyrat.algorithm import HSCC2013
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Geometry, Zonotope, cvt2, PolyZonotope
from pyrat.model import *
from pyrat.util.visualization import vis2d


def test_van_der_pol_using_zonotope():
    # init dynamic system
    system = NonLinSys.Entity(VanDerPol())

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
    results = HSCC2013.reach(system, options)

    # visualize the results
    vis2d(results, [0, 1])


def test_van_der_pol_using_polyzonotope():
    # init dynamic system
    system = NonLinSys.Entity(VanDerPol())

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
    PolyZonotope.MAX_DEPTH_GEN_ORDER = 50
    PolyZonotope.MAX_POLY_ZONO_RATIO = 0.01
    PolyZonotope.RESTRUCTURE_TECH = PolyZonotope.METHOD.RESTRUCTURE.REDUCE_PCA

    # reachable sets
    results = HSCC2013.reach(system, options)

    # visualize the results
    vis2d(results, [0, 1])


def test_tank6eq():
    # init dynamic system
    system = NonLinSys.Entity(Tank6Eq())

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

    results = HSCC2013.reach(system, options)
    vis2d(results, [0, 1])
    vis2d(results, [2, 3])
    vis2d(results, [4, 5])

import numpy as np

from pybdr.dynamic_system import NonLinSysOld
from pybdr.geometry import Zonotope, PolyZonotope
from pybdr.misc import Reachable
from pybdr.model import *
from pybdr.util.visualization import vis2d


def test_tank6Eq():
    """
    NOTE: TODO
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(Tank6Eq())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 400
    option.steps_num = 100
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([2, 4, 4, 2, 10, 4], np.eye(6) * 0)
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope([0], [[0.005]])
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])
    vis2d(reachable_results, [2, 3])
    vis2d(reachable_results, [4, 5])


def test_vanDerPol():
    """
    NOTE: TODO
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(VanDerPol())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 13
    option.steps_num = 701
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([1.4, 1.4], np.diag([0.00, 0.00]))
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope([0], [[0.005]])
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])


def test_vanDerPol_PolyZonotope():
    """
    NOTE:
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(VanDerPol())
    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 6.74
    option.steps_num = 1349
    option.zonotope_order = 50
    option.algo = "poly"
    option.tensor_order = 3
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope.zero(1, 1)
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])


def test_laubLoomis():
    """
    NOTE: TODO
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(LaubLoomis())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 20
    option.steps_num = 500
    option.taylor_terms = 20
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45], np.eye(7) * 0)
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope.zero(1, 1)
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])
    vis2d(reachable_results, [2, 4])
    vis2d(reachable_results, [5, 6])
    vis2d(reachable_results, [1, 5])
    vis2d(reachable_results, [4, 6])


def test_linear_model():
    """
    NOTE: TODO
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(RandModel())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 50
    option.steps_num = 100
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([1.4, 1.4, 1.4], np.diag([0, 0, 0]))
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope([0], [[]])
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])
    vis2d(reachable_results, [1, 2])
    vis2d(reachable_results, [0, 2])


def test_ltv():
    """

    NOTE:
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(LTV())
    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 5
    option.steps_num = 500
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([1.25, 5.25, 0], np.eye(3) * 0)
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope([0, 0, 0, 0], np.diag([0, 0, 0, 0]))
    option.u_trans = np.zeros(4)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])


def test_genetic():
    """

    NOTE:
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(GeneticModel())
    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 3
    option.steps_num = 500
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope(
        [1.0, 1.3, 0.1, 0.1, 0.1, 1.3, 2.5, 0.6, 1.3], np.eye(9) * 0
    )
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope([0], [[]])
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [2, 4])
    vis2d(reachable_results, [3, 5])


def test_p53_small():
    """

    NOTE:
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSysOld.Sys(P53Small())
    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSysOld.OptionOld()
    option.t_end = 10
    option.steps_num = 500
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([20.0] * 6, np.eye(6) * 0)
    option.r_init[0].err = np.zeros(option.r_init[0].set.shape, dtype=float)
    option.u = Zonotope([0], [[]])
    option.u_trans = np.zeros(1)

    """
    over approximating reachability analysis -------------------------------------------
    """
    reachable_results = system.reach(option)

    """
    visualization the results
    """

    vis2d(reachable_results, [0, 1])
    vis2d(reachable_results, [2, 5])


def test_continuous_system():
    system = NonLinSysOld.Sys(VanDerPol())
    option = NonLinSysOld.Option.Linear()
    raise None

import numpy as np

from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.misc import Reachable
from pyrat.model import Tank6Eq, VanDerPol, LaubLoomis, RandModel
from pyrat.util.visualization import vis2d


def test_tank6Eq():
    """
    NOTE: TODO
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSys.Sys(Tank6Eq())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSys.Option()
    option.t_end = 400
    option.steps = 401
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([2, 4, 4, 2, 10, 4], np.eye(6) * 0.2)
    # option.r_init[0].set = VectorZonotope(
    #     np.vstack([np.array([1.4, 2.4]), np.diag([0.17, 0.006])]).T
    # )
    option.r_init[0].err = np.zeros(option.r_init[0].set.dim, dtype=float)
    # option.u = Zonotope(np.array([0, 0.005]).reshape((1, -1)))
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
    system = NonLinSys.Sys(VanDerPol())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSys.Option()
    option.t_end = 13
    option.steps = 701
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([1.4, 1.4], np.diag([0.00, 0.00]))
    option.r_init[0].err = np.zeros(option.r_init[0].set.dim, dtype=float)
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


def test_laubloomis():
    """
    NOTE: TODO
    """

    """
    init nonlinear system --------------------------------------------------------------
    """
    system = NonLinSys.Sys(LaubLoomis())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSys.Option()
    option.t_end = 5
    option.steps = 1000
    option.taylor_terms = 20
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope(
        [1.2, 1.05, 1.5, 2.4, 1, 0.1, 0.45], np.eye(7) * 0.01
    )
    option.r_init[0].err = np.zeros(option.r_init[0].set.dim, dtype=float)
    option.u = Zonotope([0], [[0]])
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
    system = NonLinSys.Sys(RandModel())

    """
    init options for the computation ---------------------------------------------------
    """
    option = NonLinSys.Option()
    option.t_end = 50
    option.steps = 100
    option.taylor_terms = 4
    option.zonotope_order = 50
    option.algo = "lin"
    option.tensor_order = 2
    option.lagrange_rem["simplify"] = "simplify"
    option.r_init = [Reachable.Element()]
    option.r_init[0].set = Zonotope([1.4, 1.4, 1.4], np.diag([0, 0, 0]))
    option.r_init[0].err = np.zeros(option.r_init[0].set.dim, dtype=float)
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

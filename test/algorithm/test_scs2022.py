import numpy as np
from numpy.polynomial import Polynomial
from pyrat.algorithm import SCS2022
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.model import *


def test_computer_based_model():
    """
    NOTE:

    test case for reach avoid analysis for stochastic discrete time system
    """
    # init dynamic system
    system = NonLinSys.Entity(ComputerBasedODE())

    # settings for the controller synthesis
    options = SCS2022.Options()
    options.u = Zonotope([0], [[5]])
    options.x0 = np.zeros(system.dim)
    options.step = 0.01
    options.target = lambda x: 10 * x[0] ** 2 + 10 * (x[1] - 0.5) ** 2 - 1
    # for sampling controller function
    options.px = lambda x, params: np.array(
        params[0] * x[0] + params[1] * x[1] + params[2], dtype=float
    ).reshape(1)
    options.n = 3
    options.low = -1
    options.up = 1

    # zonotope settings
    Zonotope.ORDER = 50
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD

    # synthesis the controller
    results = SCS2022.synthesis(system, options)
    print(len(results))

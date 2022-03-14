import numpy as np

from pyrat.dynamic_system import LinearSystem

from pyrat.dynamic_system.continuous_system.linear_system.functional import Options


def test_reach():
    xa = np.random.rand(2, 2)
    xb = np.random.rand(2, 2)
    l = LinearSystem(xa, xb, 0, xa, xb, 0)
    p = Options
    p.algo = "std"
    l.over_reach(p)
    pass

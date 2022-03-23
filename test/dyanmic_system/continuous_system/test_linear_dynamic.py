import numpy as np

from pyrat.dynamic_system import LinearSystem, LSOptions


def test_reach():
    xa = np.random.rand(2, 2)
    xb = np.random.rand(2, 2)
    sys = LinearSystem(xa, xb, 0, xa, xb, 0)
    p = LSOptions()
    p.algo = "standard"
    sys.over_reach(p)

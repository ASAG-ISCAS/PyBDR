import numpy as np

from pyrat.util.functional import CSPSolver
from pyrat.util.visualization import plot


def test_case_0():
    def f(x):
        z = 3 * x[0] ** 2 + 7 * x[1] ** 2 - 1
        ind = np.logical_and(z.inf <= 0, z.sup >= 0)
        return ind[0]

    lb, ub = np.ones(2) * -5, np.ones(2) * 5
    results = CSPSolver.solve(f, lb, ub, 0.03)

    plot(results, [0, 1])

from sympy import *
from pyrat.deprecated.model_old import ModelOld


"""
NOTE:  https://flowstar.org/benchmarks/9-dimensional-genetic-model/
"""


def _f(x, u):
    dxdt = [None] * 9

    dxdt[0] = 50 * x[2] - 0.1 * x[0] * x[5]
    dxdt[1] = 100 * x[3] - x[0] * x[1]
    dxdt[2] = 0.1 * x[0] * x[5] - 50 * x[2]
    dxdt[3] = x[1] * x[5] - 100 * x[3]
    dxdt[4] = 5 * x[2] + 0.5 * x[0] - 10 * x[4]
    dxdt[5] = 50 * x[4] + 100 * x[3] - x[5] * (0.1 * x[0] + x[1] + 2 * x[7] + 1)
    dxdt[6] = 50 * x[3] + 0.01 * x[1] - 0.5 * x[6]
    dxdt[7] = 0.5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7]
    dxdt[8] = 2 * x[5] * x[7] - x[8]

    return Matrix(dxdt)


class GeneticModel(ModelOld):
    vars = symbols(("x:9", "u:1"))
    f = _f(*vars)
    name = "Genetic"
    dim = f.rows

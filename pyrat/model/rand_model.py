from sympy import *
from pyrat.deprecated.model_old import ModelOld


def _f(x, u):
    dxdt = [None] * 3

    dxdt[0] = 0.01 * x[0] + sqrt(x[1])
    dxdt[1] = x[1] ** (-1)
    dxdt[2] = x[0] * x[1]
    return Matrix(dxdt)


class RandModel(ModelOld):
    vars = symbols(("x:3", "u:1"))
    f = _f(*vars)
    name = "Random Model"
    dim = f.rows

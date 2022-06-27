from sympy import *

from pyrat.deprecated.model_old import ModelOld

"""
NOTE: 
"""


def _f(x, u):
    dxdt = [None] * 2

    px = u[0] * x[0] + u[1] * x[1] + u[2]  # ax+by+c

    dxdt[0] = -0.5 * x[0] - 0.5 * x[1] + 0.5 * x[0] * x[1]
    dxdt[1] = -0.5 * x[1] + 1 + px

    return Matrix(dxdt)


class ComputerBasedODE(ModelOld):
    vars = symbols(("x:5", "u:1"))
    f = _f(*vars)
    name = "computer based ode"
    dim = f.rows

from sympy import *
from .model import Model

"""
NOTE: 
"""


def _f(x, u):
    dxdt = [None] * 2

    dxdt[0] = -0.5 * x[0] - 0.5 * x[1] + 0.5 * x[0] * x[1]
    dxdt[1] = -0.5 * x[1] + 1 + u[0]

    return Matrix(dxdt)


class ComputerBasedODE(Model):
    vars = symbols(("x:2", "u:1"))
    f = _f(*vars)
    name = "computer based ode"
    dim = f.rows

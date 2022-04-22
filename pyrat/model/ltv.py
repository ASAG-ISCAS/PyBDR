from sympy import *
from .model import Model

"""
NOTE: https://flowstar.org/benchmarks/2-dimensional-ltv-system/
"""


def _f(x, u):
    dxdt = [None] * 3

    dxdt[0] = -x[0] - x[2] * x[1] + x[2] + u[0] + u[2]
    dxdt[1] = (x[2] ** 2) * x[0] + x[1] - x[2] + u[1] + u[3]
    dxdt[2] = 1

    return Matrix(dxdt)


class LTV(Model):
    vars = symbols(("x:3", "u:4"))
    f = _f(*vars)
    name = "LTV"
    dim = f.rows

from sympy import *
from .model import Model

"""
NOTE: https://flowstar.org/benchmarks/p53-model/
"""


def _f(x, u):
    dxdt = [None] * 6

    dxdt[0] = (0.5 - 9.963e-6 * x[0] * x[4] - 1.925e-5 * x[0]) * 3600
    dxdt[1] = (
        1.5e-3 + 1.5e-2 * (x[0] ** 2 / (547600 + x[0] ** 2)) - 8e-4 * x[1]
    ) * 3600
    dxdt[2] = (8e-4 * x[1] - 1.444e-4 * x[2]) * 3600
    dxdt[3] = (1.66e-2 * x[2] - 9e-4 * x[3]) * 3600
    dxdt[4] = (9e-4 * x[3] - 1.66e-7 * x[3] * 2 - 9.963e-6 * x[4] * x[5]) * 3600
    dxdt[5] = (0.5 - 3.209e-5 * x[5] - 9.963e-6 * x[4] * x[5]) * 3600

    return Matrix(dxdt)


class P53Small(Model):
    vars = symbols(("x:6", "u:1"))
    f = _f(*vars)
    name = "P53Small"
    dim = f.rows

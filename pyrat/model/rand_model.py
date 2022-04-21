from sympy import *
from .model import Model


def _f(x, u):
    dxdt = [None] * 2

    dxdt[0] = 0.01 * x[0] + x[1] ** 2
    dxdt[1] = x[1] ** (-1)
    return Matrix(dxdt)


class RandModel(Model):
    vars = symbols(("x:2", "u:1"))
    f = _f(*vars)
    name = "Random Model"
    dim = f.rows

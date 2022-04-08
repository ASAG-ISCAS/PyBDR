from sympy import *
from .model import Model


def _f(_x, _u):
    mu = 1

    dxdt = [None] * 2

    dxdt[0] = _x[1]
    dxdt[1] = mu * (1 - _x[0] ** 2) * _x[1] - _x[0] + _u[0]

    return Matrix(dxdt)


class VanDerPol(Model):
    vars = symbols(("x:2", "u:1"))
    f = _f(*vars)
    name = "VanDerPol"
    dim = f.rows

from .model import Model
from sympy import *


def _f(x, u):
    # dynamic
    dxdt = [None] * 2

    dxdt[0] = x[1]
    dxdt[1] = 1
    return Matrix(dxdt)


class LinearModel(Model):
    """
    system dynamics for the tank benchmark (see Sec. VII in [1])
    :return:
    """

    vars = symbols(("x:2", "u:1"))
    f = _f(*vars)
    name = "Tank6Eq"
    dim = f.rows

from sympy import *
from .model import Model


def _f(_x, _u):
    # parameters
    k0, k1, g = 0.015, 0.01, 9.81
    # differential equations
    dxdt = [None] * 6
    dxdt[0] = _u + 0.1 + k1 * (4 - _x[5]) - k0 * sqrt(2 * g) * sqrt(_x[0])  # tank 0
    dxdt[1] = k0 * sqrt(2 * g) * (sqrt(_x[0]) - sqrt(_x[1]))  # tank 1
    dxdt[2] = k0 * sqrt(2 * g) * (sqrt(_x[1]) - sqrt(_x[2]))  # tank2
    dxdt[3] = k0 * sqrt(2 * g) * (sqrt(_x[2]) - sqrt(_x[3]))  # tank3
    dxdt[4] = k0 * sqrt(2 * g) * (sqrt(_x[3]) - sqrt(_x[4]))  # tank4
    dxdt[5] = k0 * sqrt(2 * g) * (sqrt(_x[4]) - sqrt(_x[5]))  # tank5
    return Matrix(dxdt)


class Tank6Eq(Model):
    """
    system dynamics for the tank benchmark (see Sec. VII in [1])
    :return:
    """

    vars = symbols(("x:6", "u"))
    f = _f(*vars)
    name = "Tank6Eq"

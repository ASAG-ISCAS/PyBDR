from sympy import *
from .model import Model


def __f(x, u):
    # parameters
    k0, k1, g = 0.015, 0.01, 9.81
    # dynamic
    dxdt = [None] * 6

    dxdt[0] = u[0] + 0.1 + k1 * (4 - x[5]) - k0 * sqrt(2 * g) * sqrt(x[0])
    dxdt[1] = k0 * sqrt(2 * g) * (sqrt(x[0]) - sqrt(x[1]))
    dxdt[2] = k0 * sqrt(2 * g) * (sqrt(x[1]) - sqrt(x[2]))
    dxdt[3] = k0 * sqrt(2 * g) * (sqrt(x[2]) - sqrt(x[3]))
    dxdt[4] = k0 * sqrt(2 * g) * (sqrt(x[3]) - sqrt(x[4]))
    dxdt[5] = k0 * sqrt(2 * g) * (sqrt(x[4]) - sqrt(x[5]))
    return Matrix(dxdt)


Tank6Eq = Model(__f, [6, 1])

from sympy import *
from .model import Model


def __f(x, u):
    mu = 1

    dxdt = [None] * 2

    dxdt[0] = x[1]
    dxdt[1] = mu * (1 - x[0] ** 2) * x[1] - x[0] + u[0]

    return Matrix(dxdt)


VanDerPol = Model(__f, [2, 1])

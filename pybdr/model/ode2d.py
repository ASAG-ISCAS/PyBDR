from sympy import *


def ode2d(x, u):
    dxdt = [None] * 2

    dxdt[0] = -0.5 * x[0] - 0.5 * x[1] + 0.5 * x[0] * x[1]
    dxdt[1] = -0.5 * x[1] + 1 + u[0]

    return Matrix(dxdt)

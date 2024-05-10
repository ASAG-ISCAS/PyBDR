from sympy import *


def bicycle(x, u):
    dxdt = [None] * 3

    v = 2

    dxdt[0] = u[0] * cos(x[2])
    dxdt[1] = u[0] * sin(x[2])
    dxdt[2] = u[0] * tan(u[1])

    return Matrix(dxdt)

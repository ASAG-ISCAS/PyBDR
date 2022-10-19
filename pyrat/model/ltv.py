from sympy import *

"""
NOTE: https://flowstar.org/benchmarks/2-dimensional-ltv-system/
"""


def ltv(x, u):
    dxdt = [None] * 3

    dxdt[0] = -x[0] - x[2] * x[1] + x[2] + u[0] + u[2]
    dxdt[1] = (x[2] ** 2) * x[0] + x[1] - x[2] + u[1] + u[3]
    dxdt[2] = 1

    return Matrix(dxdt)

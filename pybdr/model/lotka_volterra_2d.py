from sympy import *

"""
NODE: Chen, X. (2015). Reachability analysis of non-linear hybrid systems using taylor models (Doctoral dissertation, Fachgruppe Informatik, RWTH Aachen University).
"""


def lotka_volterra_2d(x, u):
    dxdt = [None] * 2

    dxdt[0] = 1.5 * x[0] - x[0] * x[1]
    dxdt[1] = -3 * x[1] + x[0] * x[1]

    return Matrix(dxdt)

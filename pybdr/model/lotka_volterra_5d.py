from sympy import *

"""
NOTE: Chen, X. (2015). Reachability analysis of non-linear hybrid systems using taylor models (Doctoral dissertation, Fachgruppe Informatik, RWTH Aachen University).
"""


def lotka_volterra_5d(x, u):
    dxdt = [None] * 5

    dxdt[0] = x[0] * (1 - (x[0] + 0.85 * x[1] + 0.5 * x[4]))
    dxdt[1] = x[1] * (1 - (x[1] + 0.85 * x[2] + 0.5 * x[0]))
    dxdt[2] = x[2] * (1 - (x[2] + 0.85 * x[3] + 0.5 * x[1]))
    dxdt[3] = x[3] * (1 - (x[3] + 0.85 * x[4] + 0.5 * x[2]))
    dxdt[4] = x[4] * (1 - (x[4] + 0.85 * x[0] + 0.5 * x[3]))

    return Matrix(dxdt)

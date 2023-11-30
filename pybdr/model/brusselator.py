from sympy import *

"""
NOTE: Chen, X. (2015). Reachability analysis of non-linear hybrid systems using taylor models (Doctoral dissertation, Fachgruppe Informatik, RWTH Aachen University).
"""


def brusselator(x, u):
    dxdt = [None] * 2

    A = 1
    B = 1.5

    dxdt[0] = A + x[0] ** 2 * x[1] - B * x[0] - x[0]
    dxdt[1] = B * x[0] - x[0] ** 2 * x[1]

    return Matrix(dxdt)

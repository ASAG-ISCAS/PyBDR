from sympy import *

"""
NOTE: Chen, X. (2015). Reachability analysis of non-linear hybrid systems using taylor models (Doctoral dissertation, Fachgruppe Informatik, RWTH Aachen University).
"""


def jet_engine(x, u):
    dxdt = [None] * 2

    dxdt[0] = -x[1] - 1.5 * x[0] ** 2 - 0.5 * x[0] ** 3 - 0.5
    dxdt[1] = 3 * x[0] - x[1]

    return Matrix(dxdt)

from sympy import *

"""
NOTE: Chen, X. (2015). Reachability analysis of non-linear hybrid systems using taylor models (Doctoral dissertation, Fachgruppe Informatik, RWTH Aachen University).
"""


def rossler_attractor(x, u):
    dxdt = [None] * 3

    a = 0.2
    b = 0.2
    c = 5.7

    dxdt[0] = -x[1] - x[2]
    dxdt[1] = x[0] + a * x[1]
    dxdt[2] = b + x[2] * (x[0] - c)

    return Matrix(dxdt)

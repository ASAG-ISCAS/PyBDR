from sympy import *

"""
NOTE: Chen, X. (2015). Reachability analysis of non-linear hybrid systems using taylor models (Doctoral dissertation, Fachgruppe Informatik, RWTH Aachen University).
"""


def lorentz(x, u):
    dxdt = [None] * 3

    sigma = 10
    rho = 8 / 3
    beta = 28

    dxdt[0] = sigma * (x[1] - x[0])
    dxdt[1] = x[0] * (rho - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - beta * x[2]

    return Matrix(dxdt)

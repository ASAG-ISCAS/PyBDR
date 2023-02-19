from sympy import *

"""
NOTE: Chen, X. (2015). Reachability analysis of non-linear hybrid systems using taylor models (Doctoral dissertation, Fachgruppe Informatik, RWTH Aachen University).
"""


def pi_controller_with_disturbance(x, u):
    dxdt = [None] * 2

    dxdt[0] = -0.101 * (x[0] - 20) + 1.3203 * (x[1] - 0.1616) - 0.01 * x[0] ** 2
    dxdt[1] = -(-0.101 * (x[0] - 20) + 1.3203 * (x[1] - 0.1616) - 0.01 * x[0] ** 2) - 3 * (20 * x[0]) + u[0]

    return Matrix(dxdt)

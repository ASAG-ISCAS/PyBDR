from sympy import *
from .model_old import ModelOld


def _f(x, u):
    dxdt = [None] * 7

    dxdt[0] = 1.4 * x[2] - 0.9 * x[0]
    dxdt[1] = 2.5 * x[4] - 1.5 * x[1]
    dxdt[2] = 0.6 * x[6] - 0.8 * x[2] * x[1]
    dxdt[3] = 2.0 - 1.3 * x[3] * x[2]
    dxdt[4] = 0.7 * x[0] - 1.0 * x[3] * x[4]
    dxdt[5] = 0.3 * x[0] - 3.1 * x[5]
    dxdt[6] = 1.8 * x[5] - 1.5 * x[6] * x[1]

    return Matrix(dxdt)


class LaubLoomis(ModelOld):
    vars = symbols(("x:7", "u:1"))
    f = _f(*vars)
    name = "LaubLoomis"
    dim = f.rows

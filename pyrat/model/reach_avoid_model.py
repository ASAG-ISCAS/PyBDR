from sympy import *

from .model_old import ModelOld


def g(x):
    return x


def _f(x):
    u = g(x)


class ReachAvoid(ModelOld):
    vars = symbols("x:2")
    f = _f(vars)
    name = "ReachAvoid"
    dim = f.rows
   
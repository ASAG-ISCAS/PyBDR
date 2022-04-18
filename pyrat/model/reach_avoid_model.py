from sympy import *

from .model import Model


def g(x):
    return x


def _f(x):
    u = g(x)


class ReachAvoid(Model):
    vars = symbols("x:2")
    f = _f(vars)
    name = "ReachAvoid"
    dim = f.rows

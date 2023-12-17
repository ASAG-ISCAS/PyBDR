from __future__ import annotations
import numpy as np
from pybdr.model import Model


class NonLinSys:
    def __init__(self, m: Model):
        self.model = m

    @property
    def dim(self):
        return self.model.dim

    @property
    def type(self):
        return "nonlinear"

    def reverse(self):
        self.model.reverse()

    def evaluate(self, xs: tuple, mod: str, order: int, v: int):
        return self.model.evaluate(xs, mod, order, v)

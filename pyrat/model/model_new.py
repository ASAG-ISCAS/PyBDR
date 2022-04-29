from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import sympy
from sympy import symbols, Matrix


@dataclass
class ModelNEW(ABC):
    f: Matrix = None
    vars: tuple = None
    name: str = "DYNAMIC"
    dim: int = -1
    __inr_x = None
    __inr_f = None

    def __validation(self):
        univariate = isinstance(self.vars[0], sympy.Symbol)
        vars_num = 1 if univariate else len(self.vars)
        self.dim = self.f.rows
        var_dims = [len(self.vars)] if univariate else [len(v) for v in self.vars]
        assert self.f.rows == var_dims[0]
        idx = np.zeros((vars_num, 2), dtype=int)
        idx[:, 0] = np.cumsum(var_dims) - var_dims
        idx[:, 1] = np.cumsum(var_dims)
        self.__inr_x = symbols("inr_x:" + str(sum(var_dims)))
        self.__inr_f = self.f
        if not univariate:
            for var_idx in range(vars_num):
                d = dict(
                    zip(
                        self.vars[var_idx],
                        self.__inr_x[idx[var_idx, 0] : idx[var_idx, 1]],
                    )
                )
                self.__inr_f = self.__inr_f.subs(d)

    def __evaluate(self):
        raise NotImplementedError  # TODO

    def __post_init__(self):
        self.__validation()

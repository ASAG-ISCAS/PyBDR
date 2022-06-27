import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import sympy
from sympy import symbols, Matrix, derive_by_array, lambdify, ImmutableDenseNDimArray


@dataclass
class ModelNEW(ABC):
    f: Matrix = None
    vars: tuple = None
    name: str = "DYNAMIC"
    dim: int = -1
    __inr_x = None
    __inr_f = None
    __inr_series = {}

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
        self.__inr_series[0] = {"symbolic": self.__inr_f}

    def __post_init__(self):
        self.__validation()

    def __derivative(self, order: int, mod: str):
        return self.__inr_series[order][mod]

    def __take_derivative(self, order: int):
        if order - 1 not in self.__inr_series:
            self.__take_derivative(order - 1)
        d = derive_by_array(self.__derivative(order - 1, "symbolic"), self.__inr_x)
        d = np.asarray(d)
        self.__inr_series[order] = {"symbolic": np.moveaxis(d, 0, -2)}

    def evaluate(self, xs: tuple, mod: str, order: int, functional: dict = None):
        assert order >= 0
        if order not in self.__inr_series:
            self.__take_derivative(order)
        if mod not in self.__inr_series[order]:
            d = self.__derivative(order, "symbolic")
            d = d if order == 0 else d.squeeze(axis=-1)
            d = ImmutableDenseNDimArray(d)
            modules = [mod] if functional is None else functional
            self.__inr_series[order][mod] = lambdify(self.__inr_x, d, modules)

        def _eval_numpy():
            return np.asarray(self.__inr_series[order][mod](*np.concatenate(xs)))

        def _eval_interval():
            raise NotImplementedError

        if mod == "numpy":
            return _eval_numpy()
        elif mod == "interval":
            raise NotImplementedError
        else:
            raise NotImplementedError

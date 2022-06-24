import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sympy import symbols, Matrix, lambdify, derive_by_array, ImmutableDenseNDimArray


@dataclass
class ModelRef:
    f: Callable[..., Matrix] = None
    var_dims: [int] = None
    name: str = "DYNAMIC SYSTEM"
    __inr_vars = None
    __inr_x = None
    __inr_idx: np.ndarray = None
    __inr_dim: int = 0
    __inr_f: Matrix = None
    __inr_series = {}

    def __validation(self):
        vars = inspect.getfullargspec(self.f).args
        vars_num = len(vars)
        assert len(self.var_dims) == vars_num
        self.__inr_dim = sum(self.var_dims)
        assert type(self.__inr_dim) == int  # ensure input dimensions are integers
        self.__inr_vars = symbols(
            [vars[i] + ":" + str(self.var_dims[i]) for i in range(vars_num)]
        )
        self.__inr_x = symbols("inr_x:" + str(self.__inr_dim))
        self.__inr_f = self.f(*self.__inr_vars)
        self.__inr_idx = np.zeros((vars_num, 2), dtype=int)
        self.__inr_idx[:, 0] = np.cumsum(self.var_dims) - self.var_dims
        self.__inr_idx[:, 1] = np.cumsum(self.var_dims)
        for var_idx in range(len(vars)):
            start, end = self.__inr_idx[var_idx]
            d = dict(
                zip(
                    self.__inr_vars[var_idx],
                    self.__inr_x[start:end],
                )
            )
            self.__inr_f = self.__inr_f.subs(d)
        self.__inr_series[0] = {
            "sym": {v: np.asarray(self.__inr_f) for v in range(vars_num)}
        }

    def __post_init__(self):
        self.__validation()

    def __series(self, order: int, mod: str, v: int):
        return self.__inr_series[order][mod][v]

    def __take_derivative(self, order: int, v: int):
        if (
            order - 1 not in self.__inr_series
            or v not in self.__inr_series[order - 1]["sym"]
        ):
            self.__take_derivative(order - 1, v)
        start, end = self.__inr_idx[v]
        x = self.__inr_x[start:end]
        d = derive_by_array(self.__series(order - 1, "sym", v), x)
        d = np.asarray(d)
        self.__inr_series[order] = {"sym": {v: np.moveaxis(d, 0, -2)}}

    def evaluate(self, xs: tuple, mod: str, order: int, v: int):
        assert order >= 0 and 0 <= v < len(self.__inr_vars)
        if order not in self.__inr_series or v not in self.__inr_series[order]["sym"]:
            self.__take_derivative(order, v)

        def _eval_numpy():
            if mod not in self.__inr_series[order]:
                d = self.__series(order, "sym", v)
                d = d if order == 0 else d.squeeze(axis=-1)
                d = ImmutableDenseNDimArray(d)
                self.__inr_series[order][mod] = {v: lambdify(self.__inr_x, d, "numpy")}
            return np.asarray(self.__series(order, mod, v)(*np.concatenate(xs)))

        def _eval_interval():
            from pyrat.geometry import IntervalTensor

            if mod not in self.__inr_series[order]:
                d = self.__series(order, "sym", v)
                ff = np.frompyfunc(lambda x: x.is_number, 1, 1)
                xx = ff(d).astype(dtype=bool)
                mask = xx == 0
                sym_d = ImmutableDenseNDimArray(d[mask])
                vf = lambdify(self.__inr_x, sym_d, IntervalTensor.functional())
                self.__inr_series[order][mod] = {v: [vf, mask]}
            d = self.__series(order, "sym", v)
            vm = self.__series(order, mod, v)
            lb = np.zeros_like(d, dtype=float)
            ub = np.zeros_like(d, dtype=float)
            # calculate interval expressions
            vx = np.asarray(
                vm[0](
                    *[
                        xs[i][j]
                        for i in range(len(self.var_dims))
                        for j in range(self.var_dims[i])
                    ]
                )
            )
            inff = np.frompyfunc(lambda x: x.inf, 1, 1)
            supf = np.frompyfunc(lambda x: x.sup, 1, 1)
            lb[vm[1]] = inff(vx)
            ub[vm[1]] = supf(vx)
            # set remain constant values
            inv_mask = np.logical_not(vm[1])
            lb[inv_mask] = d[inv_mask].astype(dtype=float)
            ub[inv_mask] = d[inv_mask].astype(dtype=float)
            # finally return the result as interval tensor
            return IntervalTensor(lb, ub)

        if mod == "numpy":
            return _eval_numpy()
        elif mod == "interval":
            return _eval_interval()
        else:
            raise NotImplementedError

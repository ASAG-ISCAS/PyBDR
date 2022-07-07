from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
from enum import IntEnum
from pyrat.geometry.interval import Interval
from .geometry import Geometry


class ZonoTensor(Geometry.Base):
    class METHOD:
        class REDUCE(IntEnum):
            GIRARD = 0

    REDUCE_METHOD = METHOD.REDUCE.GIRARD
    MAX_ORDER = 50
    ERROR_ORDER = 20
    INTERMEDIATE_ORDER = 50

    def __init__(self, c: ArrayLike, gen: ArrayLike):
        c = c if isinstance(c, np.ndarray) else np.asarray(c, dtype=float)
        gen = gen if isinstance(gen, np.ndarray) else np.asarray(gen, dtype=float)
        assert c.shape == gen.shape[:-1] and len(c.shape) + 1 == len(gen.shape)
        self._c = c
        self._gen = gen
        self._type = Geometry.TYPE.ZONOTOPE

    @property
    def c(self):
        return self._c

    @property
    def gen(self):
        return self._gen

    @property
    def shape(self):
        return self._c.shape

    @property
    def gen_num(self):
        return self._gen.shape[-1]

    @property
    def is_empty(self):
        return np.isnan(self._c) | np.any(np.isnan(self._gen), axis=-1)

    @property
    def type(self) -> Geometry.TYPE:
        return self._type

    @property
    def T(self):
        return self.transpose()

    @property
    def R(self):
        return self.reduce(ZonoTensor.METHOD.REDUCE.GIRARD, ZonoTensor.MAX_ORDER)

    # =============================================== operations
    def __getitem__(self, item):
        c, gen = self.c[item], self.gen[item, :]
        c = c if isinstance(c, np.ndarray) else [c]
        return ZonoTensor(c, gen)

    def __setitem__(self, key, value):
        def _setitem_by_zono(x: ZonoTensor):
            self._c[key] = x.c
            self._gen[key, :] = x.gen

        def _setitem_by_number(x: (Real, np.ndarray)):
            self._c[key] = x
            self._gen[key, :] = 0

        if isinstance(value, (Real, np.ndarray)):
            return _setitem_by_number(value)
        elif isinstance(value, ZonoTensor):
            return _setitem_by_zono(value)
        else:
            raise NotImplementedError

    def __abs__(self):
        return ZonoTensor(abs(self.c), abs(self.gen))

    def __add__(self, other):
        def _add_number(x: (Real, np.ndarray)):
            return ZonoTensor(self.c + x, self.gen)

        def _add_zono(x: ZonoTensor):
            if len(self.shape) >= len(x.shape):
                xgen = np.broadcast_to(x.gen, np.append(self.shape, x.gen_num))
                gen = np.concatenate([self.gen, xgen], axis=-1)
                return ZonoTensor(self.c + x.c, gen)
            else:
                return other + self

        if isinstance(other, (Real, np.ndarray)):
            return _add_number(other)
        elif isinstance(other, ZonoTensor):
            return _add_zono(other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):

        if isinstance(other, (Real, np.ndarray)):
            return self + (-other)
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        return self - other

    def __mul__(self, other):
        def _mul_number(x: (Real, np.ndarray)):
            return ZonoTensor(self.c * x, self.gen * x)

        def _mul_interval(x: Interval):
            if len(self.shape) >= len(x.shape):
                xgen = np.broadcast_to((x.sup - x.inf) * 0.5, self.shape)
                c = self.c * x.c
                gen0 = x.c * self.gen
                gen1 = xgen * self.gen
                gen = np.concatenate([gen0, gen1], axis=-1)
                return ZonoTensor(c, gen)
            else:
                return x * self

        def _mul_zono(x: ZonoTensor):
            if len(self.shape) >= len(x.shape):
                xgen = np.broadcast_to(x.gen, np.append(self.shape, x.gen_num))
                xc = np.broadcast_to(x.c, self.shape)
                c = self.c * x.c
                gen0 = self.c[..., None] * xgen
                gen1 = xc[..., None] * self.gen
                gen2 = np.outer(self.gen, xgen).reshape(np.append(self.shape, -1))
                gen = np.concatenate([gen0, gen1, gen2], axis=-1)
                return ZonoTensor(c, gen)
            else:
                return x * self

        if isinstance(other, (Real, np.ndarray)):
            return _mul_number(other)
        elif isinstance(other, Interval):
            return _mul_interval(other)
        elif isinstance(other, ZonoTensor):
            return _mul_zono(other)
        raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __matmul__(self, other):
        raise NotImplementedError

    # =============================================== static method
    @staticmethod
    def empty(shape):
        c = np.full(shape, np.nan, dtype=float)
        gen = np.full(np.append(shape, 0), dtype=float)
        return ZonoTensor(c, gen)

    @staticmethod
    def rand(gen_num, *shape):
        assert len(shape) >= 1
        c = np.random.rand(*shape)
        gen = np.random.rand(*np.append(c.shape, gen_num))
        return ZonoTensor(c, gen)

    @staticmethod
    def zeros(gen_num, *shape):
        assert len(shape) >= 0
        c = np.zeros(shape, dtype=float)
        gen = np.zeros(np.append(c.shape, gen_num), dtype=float)
        return ZonoTensor(c, gen)

    @staticmethod
    def ones(gen_num, *shape):
        assert len(shape) >= 0
        c = np.ones(shape, dtype=float)
        gen = np.zeros(np.append(c.shape, gen_num), dtype=float)
        return ZonoTensor(c, gen)

    # =============================================== public method

    def transpose(self, axis=None):
        if axis is None:
            s = range(self._gen.ndim)[::-1]  # TODO
            s = np.roll(s, -1)
            return ZonoTensor(self._c.transpose(), self._gen.transpose(s))
        return ZonoTensor(self._c.transpose(*axis), self._gen.transpose(*axis))

    def sum(self, axis=None):
        c = self._c.sum(axis)
        return ZonoTensor(c, self._gen.reshape(np.append(c.shape, self._gen.shape[-1])))

    def proj(self, dims):
        return ZonoTensor(self.c[dims], self.gen[dims])

    def reduce(self, method: REDUCE_METHOD, order: int):
        def __reduce_girard():
            # TODO
            raise NotImplementedError

        raise NotImplementedError

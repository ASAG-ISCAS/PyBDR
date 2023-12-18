import numpy as np

"""
x(t)' = A*x(t) + B*u(t)
"""


class LinearSystemSimple:
    def __init__(self, xa, ub):
        self._xa = np.atleast_2d(xa)
        self._ub = np.atleast_2d(ub)
        assert self._xa.shape[1] == self._ub.shape[0]

    @property
    def dim(self):
        return self._xa.shape[1]

    @property
    def type(self):
        return 'linear_simple'

    @property
    def xa(self):
        return self._xa

    @property
    def ub(self):
        return self._ub

    def evaluate(self, x: np.ndarray, u: np.ndarray):
        return (self._xa[..., :, :] @ x[:, :, None] + self._ub[..., :, :] @ u[:, :, None]).squeeze(axis=-1)

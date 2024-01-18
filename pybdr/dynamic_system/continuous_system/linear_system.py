import numpy as np

"""
x(t)' = A*x(t) + B*u(t)
"""


class LinSys:
    def __init__(self, xa, ub=None):
        self._xa = np.atleast_2d(xa)
        if ub is not None:
            self._ub = np.atleast_2d(ub)
            assert self._xa.shape[1] == self._ub.shape[0]
        else:
            self._ub = None

    @property
    def dim(self):
        return self._xa.shape[1]

    @property
    def type(self):
        return 'linear'

    @property
    def xa(self):
        return self._xa

    @property
    def ub(self):
        return self._ub

    def evaluate(self, x: np.ndarray, u: np.ndarray):
        return (self._xa[..., :, :] @ x[:, :, None] + self._ub[..., :, :] @ u[:, :, None]).squeeze(axis=-1)

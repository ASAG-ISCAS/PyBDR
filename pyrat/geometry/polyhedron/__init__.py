from __future__ import annotations

import numpy as np

from .functional._mks_sub import _mks_sub
from .functional._mks_add import _mks_add


class Polyhedron:
    from .functional import (
        _init_from_hrep,
        _init_from_vrep,
        _compute_min_vrep,
        _compute_min_hrep,
        dim,
        has_hrep,
        has_vrep,
        compute_hrep,
        compute_vrep,
        eqa,
        eqb,
        ieqa,
        ieqb,
        eqh,
        ieqh,
        r,
        v,
        irr_vrep,
        irr_hrep,
    )

    # set properties
    dim = property(dim)
    has_hrep = property(has_hrep)
    has_vrep = property(has_vrep)
    eqa = property(eqa)
    eqb = property(eqb)
    eqh = property(eqh)
    ieqh = property(ieqh)
    ieqa = property(ieqa)
    ieqb = property(ieqb)
    r = property(r)
    v = property(v)
    irr_hrep = property(irr_hrep)
    irr_vrep = property(irr_vrep)

    def __init__(self, arr: np.ndarray, opt: str = "h"):
        self.__pre_init()
        if opt == "h":
            # init from halfspace representation as inequality constraints a@x<=b
            self._init_from_hrep(arr)
            self.__post_init()
        elif opt == "v":
            # init from vertices as matrix storing column vectors
            self._init_from_vrep(arr)
            self.__post_init()
        else:
            raise Exception("Unsupported initialization")

    def __pre_init(self):
        """
        declare necessary variables
        :return:
        """
        self._zo_tol = 1e-15  # tolerance for nearly-zero checking
        self._region_tol = 1e-7  # region with diameter less than the one of inscribed ball(twice the Chebyshev radius) are considered as empty
        self._abs_tol = 1e-8  # rank, general comparisons <,>, convergence, inversion
        # internal variables for specific property checking
        self._int_empty = None
        self._int_bounded = None
        self._int_fullspace = None
        self._int_inner_pt = None
        self._int_cheby_data = None
        self._int_lb = None
        self._int_ub = None
        # necessary variables relates to constraints define a polyhedron
        self._ieqh = None
        self._eqh = None
        self._v = None
        self._r = None
        self._irr_hrep = False
        self._irr_vrep = False
        self._has_hrep = False
        self._has_vrep = False
        self._dim = -1

    def __post_init(self):
        # TODO
        pass

    # ------------------------------- numeric operators
    def __add__(self, other):
        return _mks_add(self, other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _mks_sub(self, other)

    def __isub__(self, other):
        return self.__sub__(other)

    # ------------------------------- static methods
    @staticmethod
    def empty(dim: int) -> Polyhedron:
        """
        construct an empty set in R^n
        :param dim: dimension of the target space
        :return: An empty polyhedron in target dimensional space
        """
        p = Polyhedron(np.zeros((0, dim + 1), dtype=float))
        p._int_empty = True
        p._int_fullspace = False
        p._int_lb = np.full((1, dim), np.inf, dtype=float)
        p._int_ub = np.full((dim, 1), -np.inf, dtype=float)
        return p

    @staticmethod
    def fullspace(dim: int) -> Polyhedron:
        """
        construct the H-representation of R^n
        :param dim: dimension of the target space
        :return: A full-dimensional polyhedron
        """
        # R^n is represented by 0@x<=1
        h = np.zeros((1, dim + 1), dtype=float)
        h[:, -1] = 1
        p = Polyhedron(h)
        p._irr_hrep = True
        p._int_empty = dim == 0  # R^0 is an empty set
        p._int_fullspace = dim > 0  # R^0 is not fully dimensional
        p._int_bounded = dim == 0  # R^0 is bounded
        p._int_lb = np.full((dim, 1), -np.inf, dtype=float)
        p._int_ub = np.full((dim, 1), np.inf, dtype=float)
        return p

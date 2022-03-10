from __future__ import annotations
import numpy as np


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
        _mks_add,
        _mks_sub,
        empty,
        fullspace,
        __add__,
        __iadd__,
        __sub__,
        __isub__,
        is_empty,
        is_fullspace,
        is_fulldim,
        is_bounded,
        contains,
        lrs,
        removed_halfspaces,
    )

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

    @classmethod
    def _new(cls, arr: np.ndarray, opt: str) -> Polyhedron:
        return Polyhedron(arr, opt)

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
        self._int_fulldim = None
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

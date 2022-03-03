from __future__ import annotations

import numpy as np


class Polyhedron:
    from .functional import (
        min_affine_rep,
        empty,
        fullspace,
        _init_from_hrep,
        _init_from_vrep,
        _compute_min_vrep,
        _compute_min_hrep,
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

    def __pre_init(self):
        """
        declare necessary variables
        :return:
        """
        self._zo_tol = 1e-15  # tolerance for nearly-zero checking
        # internal variables for specific property checking
        self._int_empty = []
        self._int_bounded = []
        self._int_inner_pt = []
        self._int_cheby_data = []
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

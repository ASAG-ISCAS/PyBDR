from __future__ import annotations

import numpy as np
from .functional import *
from pyrat.util.functional.aux import *


class Polyhedron:
    def __init__(self, arr: np.ndarray, opt: str = "h"):
        self.__zero_tol = 1e-15  # tolerance for nearly-zero checking
        self.__pre_init()
        if opt == "h":
            # init from halfspace representation as inequality constraints A*x<=b
            self.__init_from_hrep(arr)
            self.__post_init()
        elif opt == "v":
            # init from vertex representation
            self.__init_from_vrep(arr)
            self.__post_init()
        else:
            raise Exception("unsupported initialization")

    def __pre_init(self):
        """
        necessary definition before initialization, define necessary basic variables mostly
        :return:
        """
        # internal variable for specific property checking purposes
        self.__init_empty = []
        self.__init_bounded = []
        self.__init_inner_pt = []
        self.__init_cheby_data = []
        # necessary variables relates to constraints define a polyhedron
        self.__ieqh = None
        self.__eqh = None
        self.__v = None
        self.__r = None
        self.__irr_hrep = False
        self.__irr_vrep = False
        self.__has_hrep = False
        self.__has_vrep = False
        self.__dim = -1

    def __post_init(self):
        """
        necessary definition after initialization
        :return:
        """
        self.__normalize()
        # TODO do normalization for now

    def __init_from_hrep(self, hrep: np.ndarray):
        """
        init polyhedron from hrep as linear inequalities
        :param hrep: linear inequalities hrep=[m,v] indicates m*x<=v
        :return:
        """
        assert hrep.shape[0] == hrep.shape[1] - 1
        # replace c'*x<=+/-inf by 0'*x<=+/-1
        inf_rows = np.isinf(hrep[:, -1])
        if any(inf_rows):
            hrep[inf_rows, :-1] = 0
            hrep[inf_rows, -1] = np.sign(hrep[:, -1])
        # replace nearly-zero entries by zero
        hrep[abs(hrep) < self.__zero_tol] = 0
        self.__dim = hrep.shape[0]
        self.__ieqh = hrep
        self.__eqh = np.zeros((0, self.dim + 1), dtype=float)
        self.__v = np.zeros((0, self.dim))
        self.__r = np.zeros((0, self.dim))
        self.__has_hrep = not is_empty(self.ieqh)

    def __init_from_vrep(self, vrep: np.ndarray):
        assert not is_empty(vrep)
        # replace nearly-zero entries by zero
        vrep[abs(vrep) < self.__zero_tol] = 0
        self.__dim = vrep.shape[1]
        self.__v = vrep
        self.__ieqh = np.zeros((0, self.dim + 1), dtype=float)
        self.__eqh = np.zeros((0, self.dim + 1), dtype=float)
        self.__r = np.zeros((0, self.dim))
        self.__has_hrep = not is_empty(self.ieqh) or not is_empty(self.eqh)
        self.__has_vrep = not is_empty(self.v) or not is_empty(self.r)
        # compute minimum representation for the affine set
        if not is_empty(self.eqh):
            if np.linalg.norm(self.eqh) == 0 and (
                    is_empty(self.ieqh) or np.linalg.norm(self.ieqh) == 0
            ):
                # corner case 0*x=0
                h = np.zeros((1, self.dim + 1), dtype=float)
                h[:, -1] = 1
                self.__init_from_hrep(h)
                self.__irr_hrep = True  # full space representation
            else:
                self.__eqh = min_affine_rep(self.eqh)

    def __compute_hrep(self):
        if self.has_hrep:
            return  # do nothing
        elif not self.has_vrep:
            # empty set

            # TODO
            pass
        # TODO
        pass

    def __compute_vrep(self):
        # TODO
        pass

    def __normalize(self):
        """
        normalize polyhedron such that each facet i a_i.T@x<=b_i is scaled with ||a_i||_2=1
        :return:
        """

        def __normalize_hrep(hrep: np.ndarray) -> np.ndarray:
            ret_hrep = hrep
            # 2-norm of each facet
            norm = np.linalg.norm(hrep[:, :-1], ord=2, axis=1)
            # normalize 0'@x<=+/-b to 0'@x<=+/- sign(b)
            zo_rows = norm < self.__zero_tol
            norm[zo_rows] = 1
            ret_hrep[:, -1][zo_rows] = np.sign(hrep[:, -1])
            # normalize each halfspace (0@x<=b will be left intact)
            ret_hrep /= norm[:, None]
            return ret_hrep

        if self.has_hrep:
            self.__ieqh = __normalize_hrep(self.ieqh)
            # normalize equalities if present
            if not is_empty(self.eqh):
                self.__eqh = __normalize_hrep(self.eqh)

    # =============================== properties
    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def irr_vrep(self) -> bool:
        return self.__irr_vrep

    @property
    def irr_hrep(self) -> bool:
        return self.__irr_hrep

    @property
    def eqh(self) -> np.ndarray:
        return self.__eqh

    @property
    def ieqh(self) -> np.ndarray:
        return self.__ieqh

    @property
    def v(self) -> np.ndarray:
        return self.__v

    @property
    def r(self) -> np.ndarray:
        return self.__r

    @property
    def has_hrep(self) -> bool:
        return self.__has_hrep

    @property
    def has_vrep(self) -> bool:
        return self.__has_vrep

    @property
    def ieqa(self) -> np.ndarray:
        if not self.has_hrep and self.has_vrep:
            # TODO
            pass
        pass

    @property
    def eqa(self) -> np.ndarray:
        # TODO
        pass

    @property
    def ieqb(self) -> np.ndarray:
        # TODO
        pass

    @property
    def eqb(self) -> np.ndarray:
        # TODO
        pass

    @property
    def is_full_dim(self) -> bool:
        # TODO
        pass

    @property
    def is_empty(self) -> bool:
        # TODO
        pass

    # =============================== static methods
    @staticmethod
    def empty(dim: int) -> Polyhedron:
        """
        construct an empty set in R^n
        :param dim:
        :return:
        """
        return Polyhedron(np.zeros((0, dim + 1), dtype=float))

    @staticmethod
    def full_space(dim: int) -> Polyhedron:
        """
        construct the H-representation of R^n which represented by 0'*x<=1
        :param dim: dimension of the target space
        :return:
        """
        h = np.zeros((1, dim + 1), dtype=float)
        h[:, -1] = 1
        return Polyhedron(h)

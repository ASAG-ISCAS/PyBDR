from __future__ import annotations

import numpy as np
from .functional import *
from pyrat.util.functional.aux import *


class Polyhedron:
    def __init__(self, arr: np.ndarray, opt: str = 'h'):
        self.__zero_tol = 1e-15  # tolerance for nearly-zero checking
        if opt == 'h':
            # init from halfspace representation as inequality constraints A*x<=b
            self.__init_from_hrep(arr)
        elif opt == 'v':
            # init from vertex representation
            self.__init_from_vrep(arr)
        else:
            raise Exception('unsupported initialization')

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
        self.__ieqH = hrep
        self.__eqH = np.zeros((0, self.dim + 1), dtype=float)
        self.__v = np.zeros((0, self.dim))
        self.__r = np.zeros((0, self.dim))
        self.__irr_hrep = False
        self.__irr_vrep = False

    def __init_from_vrep(self, vrep: np.ndarray):
        assert not is_empty(vrep)
        # replace nearly-zero entries by zero
        vrep[abs(vrep) < self.__zero_tol] = 0
        self.__dim = vrep.shape[1]
        self.__v = vrep
        self.__ieqH = np.zeros((0, self.dim + 1), dtype=float)
        self.__eqH = np.zeros((0, self.dim + 1), dtype=float)
        self.__r = np.zeros((0, self.dim))
        self.__irr_vrep = False
        self.__irr_hrep = False
        self.__compute_min_rep()

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

    def __compute_min_rep(self):
        # compute minimum representation for the affine set
        if not is_empty(self.eqH):
            if np.linalg.norm(self.eqH) == 0 and (is_empty(self.ieqH) or np.linalg.norm(self.ieqH) == 0):
                # corner case 0*x=0
                h = np.zeros((1, self.dim + 1), dtype=float)
                h[:, -1] = 1
                self.__init_from_hrep(h)
                self.__irr_hrep = True  # full space representation
            else:
                self.__eqH = min_affine_rep(self.eqH)

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
    def eqH(self) -> np.ndarray:
        return self.__eqH

    @property
    def ieqH(self) -> np.ndarray:
        return self.__ieqH

    @property
    def v(self) -> np.ndarray:
        return self.__v

    @property
    def r(self) -> np.ndarray:
        return self.__r

    @property
    def has_hrep(self) -> bool:
        return not is_empty(self.ieqH) or not is_empty(self.eqH)

    @property
    def has_vrep(self) -> bool:
        return not is_empty(self.v) or not is_empty(self.r)

    @property
    def ieqA(self) -> np.ndarray:
        if not self.has_hrep and self.has_vrep:
            pass

        pass

    @property
    def eqA(self) -> np.ndarray:
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

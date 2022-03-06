from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron
from pyrat.util.functional.aux_numpy import *
from pyrat.util.functional.aux_python import *
import cvxpy as cp


@reg_property
def is_fullspace(self: Polyhedron) -> bool:
    """
    check if this polyhedron represents R^n
    :param self: this polyhedron shall be checked
    :return:
    """
    is_full = False
    if self.has_hrep:
        if not is_empty(self.eqh):
            # affine set is not R^n
            pass
        elif (
            is_empty(self.eqh)
            and np.all(np.linalg.norm(self.ieqh[:, :-1], ord=2, axis=1) < self._zo_tol)
            and np.all(np.linalg.norm(self.ieqh[:, -1], ord=2) > -self._zo_tol)
        ):
            is_full = True
    elif self.has_vrep and not is_empty(self.r):
        # check whether rays span R^n
        """
        the rays span R^n if there exists lambda>=0 such that R*lambda gives each basic
        vector of R^n
        """
        is_full = True
        # set of basic vectors
        e = np.eye(self.dim, dtype=float)
        e = np.concatenate(e, -e, axis=0)
        r = self.r
        r_num = self.r.shape[1]

        # TODO
        pass

        # TODO
        pass

    # TODO
    pass

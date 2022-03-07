from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyrat.geometry import Polyhedron

from pyrat.util.functional.aux_numpy import *
from pyrat.util.functional.aux_python import *
from pyrat.util.functional.aux_solver import lp


@reg_property
def cheby_center(
    self: Polyhedron, facet: np.ndarray = None, bound: float = np.inf
) -> (np.ndarray, float, bool):
    """
    compute the cheby center of this polyhedron
    :param self: this polyhedron instance
    :param facet: if facet index is specified, then set related inequality to equality
    :param bound:
    :return:

    Artificial bound on the radius. Logic:
        + solve the linear programming problem with this big bound on the radius
        + if the LP is infeasible => exit, infeasible
        + if the radius is smaller than big_bound => exit, bounded polytope
        + if the radius is larger, we either have a larger polytope, or an unbounded
          polyhedron. drop the bound altogether and solve the LP again. Then:
            + if the LP without bounds on the radius is feasible => polytope
            + otherwise => unbounded polyhedron
    """
    big_bound = 1.234e6
    # check for polyhedra
    if self.has_hrep:
        if not self.has_vrep:
            # empty polyhedron
            return None, -np.inf, False
        # get the H-rep
        self._compute_min_hrep()
    ieqh = self.ieqh
    eqh = self.eqh
    # if facets provided, P must be irredundant
    if facet is not None:
        if not self.irr_hrep:
            raise OSError(
                "Polyhedron must be in minimal representation when you want compute "
                + "Chebyshev center any of its facets. Use 'computer_min_hrep()' to"
                + " compute the facets "
            )
    # if chebyshev data are stored internally, use this information
    if self._int_cheby_data is None and facet is None and np.isinf(bound):
        # only available if there is no facet and bound provided
        if self._int_cheby_data is not None:
            return self._int_cheby_data

    # Chebyshev center solves a following LP:
    """
    Chebyshev center solves a following LP:
    max r
    s.t. a_i'@xc + ||a_i||_2@r <=b_i i=1,...,m
         Ae@xc=be
    where a_i is the i-th row of the inequality matrix A in A@xc<=b and ||a_i||_2 is the 
    2-norm of that row
    """
    # inequality constraints on [xc,r]
    # [a_i, ||a_i||_2], i=1,...,m
    ieqa = ieqh[:, :-1]
    ieqa = np.concatenate([ieqa, np.sqrt((ieqa.T * ieqa).sum(axis=1))], axis=0)
    ieqb = ieqh[:, -1]
    # equality constraints
    # [a_i, ||a_i||_2],i=1,...,me
    eqa = eqh[:, :-1]
    eqa = np.concatenate([eqa, np.zeros((eqa.shape[0], 1))], axis=0)
    eqb = eqh[:, -1]
    # if we want to compute center of the facet, add this facet to equality constraints
    if facet is not None:
        if np.any(facet > ieqa.shape[0]):
            raise Exception(
                "Facet index must be less than number of inequalities "
                + str(ieqa.shape[0])
            )
        eqa = np.concatenate(
            [
                eqa,
                np.concatenate(
                    [ieqa[facet, :-1, np.zeros((facet.shape[0], 1), dtype=float)]],
                    axis=1,
                ),
            ],
            axis=0,
        )
        eqb = np.concatenate([eqb, ieqb[facet]], axis=0)
        ieqa = np.delete(ieqa, facet, axis=0)
        ieqb = np.delete(ieqb, facet, axis=0)
    # lower bounds on [xc, r]
    ieqa = np.concatenate(
        [ieqa, np.concatenate([np.zeros((1, self.dim), dtype=float), -1], axis=1)],
        axis=0,
    )
    ieqb = np.concatenate([ieqb, 0], axis=1)
    # upper bounds
    inf_bound = np.isinf(bound)
    if inf_bound:
        # always introduce at least an artificial upper bound on the radius
        bound = big_bound
    # make sure the bound on the radius is always the last constraint!
    ieqa = np.concatenate(
        [ieqa, np.concatenate([np.zeros((1, self.dim), dtype=float), 1], axis=1)],
        axis=0,
    )
    # the last value is -1 because it is maximization
    c = np.concatenate([np.zeros((ieqh.shape[1] - 1, 1), dtype=float), -1], axis=0)
    x, v = lp(c, ieqa, ieqb, eqa, eqb)
    if x is None:  # if infeasible output
        return None, -np.inf, False
    ret_x, ret_r, ret_flag = None, None, False
    if -v > self._zo_tol:
        ret_x = x[:-1]
        ret_r = x[-1]
    else:
        ret_x = x[:-1]
        ret_r = 0
    if inf_bound and ret_r >= big_bound - 1:
        """
        the radius is too large, potentially we have an unbounded polyhedron, Re-solve
        the LP without bounds to be sure.

        introduce a numerical tolerance when comparing the radius to big_bound.

        remove the bound. if we get a bounded solution, then we have a polytope. Otherwise
        we have a polyhedron.

        NOTE: as noted above, make sure the bounds are always teh last constraint!
        """
        ieqa = ieqa[:-1, :]
        ieqb = ieqb[:-1]
        """
        Note that the LP can also be infeasible. but that indicates unboundness, since 
        it was feasible with the bound included
        """
        x, v = lp(c, ieqa, ieqb, eqa, eqb)
        if x is None:
            ret_r = np.inf  # unbounded => polyhedron
        else:
            # bounded solution even without bounds on the radius => polytope
            ret_x = x[:-1]
            ret_r = x[-1]
    # check if the point is contained inside the polyhedron
    if not self.contains(ret_x):
        ret_flag = False
    # store internally if empty facet and the bound was not provided
    if is_empty(facet) and inf_bound:
        self._int_cheby_data = ret_x, ret_r, ret_flag

import cvxpy as cp


def lp(c, ieqa=None, ieqb=None, eqa=None, eqb=None, lb=None, ub=None, threads_num=8):
    """
    linear programming solves the problem as
    min c@x
    s.t. ieqa@x<=ieqb
         eqa@x==eqb
         lb<=x<=ub
    :param c: column vector represents linear part
    :param ieqa: left-hand side inequality constraint matrix
    :param ieqb: right-hand side inequality constraint vector
    :param eqa: left-hand side equality constraint matrix
    :param eqb: right-hand side equality constraint vector
    :param lb: lower bound of the solution
    :param ub: upper bound of the solution
    :param threads_num: number of threads for OPENMP support
    :return: feasible solution x or None if infeasible problem
    """
    assert threads_num > 0 and c.shape[0] > 0
    cp.set_num_threads(threads_num)
    x = cp.Variable(c.shape[0])
    cost = c.T @ x
    constraints = []
    if ieqa is not None and ieqb is not None:
        constraints.append(ieqa @ x <= ieqb)
    if eqa is not None and eqb is not None:
        constraints.append(eqa @ x == eqb)
    if lb is not None:
        constraints.append(lb <= x)
    if ub is not None:
        constraints.append(x <= ub)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=True)  # set to TRUE to enable log output
    return x.value, prob.value

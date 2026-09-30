"""
Soundness of the reachability analysis: trajectories simulated from random initial states must
stay inside the computed reachable sets.
"""

import numpy as np
from scipy.integrate import solve_ivp

from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Geometry, Interval, Zonotope
from pybdr.geometry.operation import boundary, cvt2
from pybdr.model import Model, vanderpol

X0 = Interval([1.23, 2.34], [1.57, 2.46])


def options(t_end):
    opt = ASB2008CDC.Options()
    opt.t_end = t_end
    opt.step = 0.01
    opt.tensor_order = 3
    opt.taylor_terms = 4
    opt.u = Zonotope.zero(1, 1)
    opt.u_trans = np.zeros(1)
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50
    return opt


def simulate(x0: np.ndarray, times: np.ndarray) -> np.ndarray:
    model = Model(vanderpol, [2, 1])

    def f(_, x):
        return np.asarray(model.evaluate((x, np.zeros(1)), "numpy", 0, 0), dtype=float).ravel()

    return solve_ivp(f, (0, times[-1]), x0, t_eval=times, rtol=1e-10, atol=1e-12).y.T


def max_violation(states: np.ndarray, time_point_sets: list) -> float:
    """largest constraint value a x - b of the states w.r.t. the sets, <= 0 means contained"""
    worst = -np.inf
    for x, z in zip(states, time_point_sets):
        p = cvt2(z, Geometry.TYPE.POLYTOPE)
        worst = max(worst, np.max(p.a @ x - p.b))
    return worst


def test_reach_contains_trajectories():
    opt = options(t_end=1.0)
    ri, rp = ASB2008CDC.reach(vanderpol, [2, 1], opt, cvt2(X0, Geometry.TYPE.ZONOTOPE))
    assert len(ri) == len(rp) + 1 == opt.steps_num + 1

    times = opt.step * np.arange(1, len(rp) + 1)
    rng = np.random.default_rng(0)
    for x0 in X0.inf + (X0.sup - X0.inf) * rng.uniform(size=(20, 2)):
        assert max_violation(simulate(x0, times), rp) <= 1e-9


def test_reach_parallel_on_boundary_contains_trajectories():
    opt = options(t_end=0.5)
    cells = boundary(X0, 0.1, Geometry.TYPE.INTERVAL)
    ri, rp = ASB2008CDC.reach_parallel(vanderpol, [2, 1], opt, [cvt2(c, Geometry.TYPE.ZONOTOPE) for c in cells])
    # results are indexed by time step first, then by cell
    # (opt.steps_num is only set in the worker processes)
    assert len(ri) == len(rp) + 1 == round(opt.t_end / opt.step) + 1
    assert all(len(r) == len(cells) for r in ri + rp)

    # reach_parallel does not keep the order of the cells, match them by their initial set
    initial = [cvt2(x, Geometry.TYPE.INTERVAL) for x in ri[0]]
    times = opt.step * np.arange(1, len(rp) + 1)
    rng = np.random.default_rng(1)
    for cell in cells:
        k = next(i for i, x in enumerate(initial) if np.allclose(x.inf, cell.inf) and np.allclose(x.sup, cell.sup))
        for x0 in cell.inf + (cell.sup - cell.inf) * rng.uniform(size=(3, 2)):
            assert max_violation(simulate(x0, times), [r[k] for r in rp]) <= 1e-9

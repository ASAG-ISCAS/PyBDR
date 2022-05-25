from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle

from pyrat.algorithm import SCS2022
from pyrat.geometry import Zonotope
from .value_function import vx0


def vis(trajectory, reachable_sets, dims, width=800, height=800):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    # visualize the trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1])
    ax.plot(trajectory[:, 0], trajectory[:, 1], "ro")

    # visualize the reachable sets
    for this_set in reachable_sets:
        g = this_set.proj(dims)
        p = Polygon(
            g.polygon(),
            closed=True,
            alpha=1,
            fill=False,
            linewidth=0.5,
            edgecolor="blue",
        )
        ax.add_patch(p)

    # visualize the value function
    ax.add_patch(Circle([0, 0], radius=1, color="black", fill=False))

    # visualize the target region
    ax.add_patch(Circle([0, 0.5], radius=np.sqrt(0.1), color="g", fill=False))

    # visualize the RA region
    x, y = np.ogrid[-1:1:100j, -1:1:100j]
    ax.contour(x.ravel(), y.ravel(), vx0(y, x), [0])

    # misc
    ax.autoscale_view()
    ax.axis("equal")
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))
    plt.show()


def test_computer_based_model():
    """
    NOTE:

    test case for reach avoid analysis for stochastic discrete time system
    """
    # init model
    def f(x, px: Callable[[np.ndarray], float]):
        dxdt = [None] * 2

        dxdt[0] = -0.5 * x[0] - 0.5 * x[1] + 0.5 * x[0] * x[1]
        dxdt[1] = -0.5 * x[1] + 1 + px(x)

        return dxdt

    def p0(x: np.ndarray, params: np.ndarray) -> float:
        return params[0] * x[0] ** 2 + params[1] * x[1] + 3 * params[2] * x[0] * x[1]

    def p1(x: np.ndarray, params: np.ndarray) -> float:
        return params[0]

    def distance(pts: np.ndarray) -> np.ndarray:
        # target region is bounded by a circle, so distance can get exactly as
        diffs = pts - np.array([0, 0.5])[None, :]
        return np.linalg.norm(diffs, ord=2, axis=1) + np.sqrt(0.1)

    # settings for the controller synthesis
    options = SCS2022.Options()
    options.x0 = np.array([-0.1, -0.9])
    options.step = 0.01
    options.dim = 2
    options.target = lambda x: 10 * x[0] ** 2 + 10 * (x[1] - 0.5) ** 2 - 1
    options.vx = lambda x: vx0(x[0], x[1])
    options.distance = distance
    # for sampling controller function
    options.p = p0
    options.n = 3
    options.low = -1
    options.up = 1

    # zonotope settings
    Zonotope.ORDER = 50
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD

    # synthesis the controller
    tj, rs = SCS2022.run(f, options)
    tj = np.vstack(tj)
    vis(tj, rs, [0, 1])

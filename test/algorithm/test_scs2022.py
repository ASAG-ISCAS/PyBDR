from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle

from pyrat.algorithm import SCS2022
from pyrat.geometry import Zonotope
from pyrat.util.visualization import vis2dGeo


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

    def p(x: np.ndarray, params: np.ndarray) -> float:
        return params[0] * x[0] ** 2 + params[1] * x[1] + 3 * params[2] * x[0] * x[1]

    # settings for the controller synthesis
    options = SCS2022.Options()
    options.x0 = np.array([-0.1, -0.9])
    options.step = 0.01
    options.dim = 2
    options.target = lambda x: 10 * x[0] ** 2 + 10 * (x[1] - 0.5) ** 2 - 1
    options.vx = lambda x: x[0] ** 2 + x[1] ** 2 - 1
    # for sampling controller function
    options.p = p
    options.n = 3
    options.low = -10
    options.up = 10

    # zonotope settings
    Zonotope.ORDER = 50
    Zonotope.REDUCE_METHOD = Zonotope.METHOD.REDUCE.GIRARD

    # synthesis the controller
    tj, rs = SCS2022.synthesis(f, options)
    tj = np.vstack(tj)
    vis(tj, rs, [0, 1])

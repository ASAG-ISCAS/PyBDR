from __future__ import annotations
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from pyrat.geometry import Geometry, Zonotope, Interval
from pyrat.dynamic_system import NonLinSys
from pyrat.model import vanderpol, Model
from pyrat.algorithm import ASB2008CDC

if __name__ == '__main__':
    # init dynamic system
    system = NonLinSys(Model(vanderpol, [2, 1]))

    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4
    options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]

    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    # reachable sets computation
    ti, tp0, _, _ = ASB2008CDC.reach(system, options)

    from pyrat.geometry.operation import boundary

    c = np.array([1.4, 2.4], dtype=float)
    inf = c - [0.17, 0.06]
    sup = c + [0.17, 0.06]
    box = Interval(inf, sup)

    # box=Interval()
    options.r0 = boundary(box, 1, Geometry.TYPE.ZONOTOPE)

    _, tp1, _, _ = ASB2008CDC.reach(system, options)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    px = 1 / plt.rcParams["figure.dpi"]
    width, height = 800, 800
    dims = [0, 1]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")


    def __add_zonotope(zono: "Zonotope", color, is_filled, alpha):
        g = zono.proj(dims)
        p = Polygon(
            g.polygon(),
            closed=True,
            alpha=alpha,
            fill=is_filled,
            linewidth=1,
            edgecolor=color,
            facecolor=color,
            color=color,
        )
        ax.add_patch(p)


    assert len(tp0) == len(tp1)

    # show tp0 & tp1
    for idx in range(len(tp0)):
        c = plt.cm.turbo(idx / len(tp0))
        for r in tp1[idx]:
            __add_zonotope(r, "blue", True, 1)
        for r in tp0[idx]:
            __add_zonotope(r, c, False, 0.7)

    ax.autoscale_view()
    ax.axis("equal")
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))

    plt.show()

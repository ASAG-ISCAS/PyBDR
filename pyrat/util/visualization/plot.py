import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pyrat.geometry import Geometry, Interval, Zonotope, Polytope
from itertools import chain


def __3d_plot(objs, dims: list, width: int, height: int):
    # TODO
    raise NotImplementedError


def __2d_plot(objs, dims: list, width: int, height: int):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    def __add_pts(pts: np.ndarray, color):
        assert pts.ndim == 2
        ax.scatter(pts[:, dims[0]], pts[:, dims[1]], c=color)

    def __add_interval(i: "Interval", color):
        ax.add_patch(
            Polygon(
                i.proj(dims).rectangle(),
                closed=True,
                alpha=0.7,
                fill=False,
                linewidth=0.5,
                edgecolor=color,
            )
        )

    def __add_polytope(p: "Polytope", color):
        ax.add_patch(
            Polygon(
                p.polygon(dims),
                closed=True,
                alpha=0.7,
                fill=False,
                linewidth=2,
                edgecolor=color,
            )
        )

    def __add_zonotope(zono: "Zonotope", color):
        g = zono.proj(dims)
        p = Polygon(
            g.polygon(),
            closed=True,
            alpha=1,
            fill=False,
            linewidth=1,
            edgecolor=color,
        )
        ax.add_patch(p)

    for i in range(len(objs)):
        c = plt.cm.turbo(i / len(objs))
        geos = (
            [objs[i]]
            if not isinstance(objs[i], list)
            else list(itertools.chain.from_iterable([objs[i]]))
        )
        for geo in geos:
            if isinstance(geo, np.ndarray):
                __add_pts(geo, c)
            elif isinstance(geo, Geometry.Base):
                if geo.type == Geometry.TYPE.INTERVAL:
                    __add_interval(geo, "black")
                elif geo.type == Geometry.TYPE.POLYTOPE:
                    __add_polytope(geo, c)
                elif geo.type == Geometry.TYPE.ZONOTOPE:
                    __add_zonotope(geo, c)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    ax.autoscale_view()
    ax.axis("equal")
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))

    plt.show()


def plot(objs, dims: list, mod: str = "2d", width: int = 800, height: int = 800):
    if mod == "2d":
        return __2d_plot(objs, dims, width, height)
    elif mod == "3d":
        return __3d_plot(objs, dims, width, height)
    else:
        raise Exception("unsupported visualization mode")

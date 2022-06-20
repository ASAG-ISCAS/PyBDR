import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pyrat.geometry import Geometry, Interval, Zonotope, Polytope
from pyrat.misc import Reachable


def __3d_plot(objs, dims: list, width: int, height: int):
    # TODO
    raise NotImplementedError


def __2d_plot(objs, dims: list, width: int, height: int):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    def __add_pts(pts: np.ndarray):
        assert pts.ndim == 2
        ax.scatter(pts[:, dims[:, 0]], pts[:, dims[:, 1]])

    def __add_interval(i: "Interval"):
        ax.add_patch(
            Polygon(
                i.rectangle(dims),
                closed=True,
                alpha=0.7,
                fill=False,
                linewidth=0.5,
                edgecolor="black",
            )
        )

    def __add_polytope(p: "Polytope"):
        ax.add_patch(
            Polygon(
                p.polygon(dims),
                closed=True,
                alpha=0.7,
                fill=False,
                linewidth=0.5,
                edgecolor="red",
            )
        )

    def __add_zonotope(zono: "Zonotope"):
        g = zono.proj(dims)
        p = Polygon(
            g.polygon(),
            closed=True,
            alpha=1,
            fill=False,
            linewidth=0.5,
            edgecolor="blue",
        )
        ax.add_patch(p)

    for obj in objs:
        if isinstance(obj, np.ndarray):
            __add_pts(obj)
        elif isinstance(obj, Geometry.Base):
            if obj.type == Geometry.TYPE.INTERVAL:
                __add_interval(obj)
            elif obj.type == Geometry.TYPE.POLYTOPE:
                __add_polytope(obj)
            elif obj.type == Geometry.TYPE.ZONOTOPE:
                __add_zonotope(obj)
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

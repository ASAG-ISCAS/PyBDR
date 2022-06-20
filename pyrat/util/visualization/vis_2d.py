import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle

from pyrat.geometry import Geometry
from pyrat.misc import Reachable


def vis2d(r: Reachable.Result, dims: list, width=800, height=800):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    def vis_element(g: Geometry.Base):
        g = g.proj(dims)
        p = Polygon(
            g.polygon(),
            closed=True,
            alpha=1,
            fill=False,
            linewidth=0.5,
            edgecolor="blue",
        )
        ax.add_patch(p)

    if len(r.tis) >= 0:
        vis2dGeo(r.tis, dims)
        return
        for res in r.tis:
            for re in res:
                vis_element(re)
    else:
        vis2dGeo(r.tps, dims)
        return
        for res in r.tps:
            for re in res:
                vis_element(re)

    ax.autoscale_view()
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))

    plt.show()


def vis2dGeo(geos: [Geometry.Base], dims: list, width=800, height=800):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    def __add_patch(obj: Geometry.Base):
        if isinstance(obj, np.ndarray):
            assert obj.ndim == 2
            pts = obj[:, dims]
            ax.scatter(pts[:, 0], pts[:, 1])
        elif isinstance(obj, Geometry.Base):
            if obj.type == Geometry.TYPE.INTERVAL:
                ax.add_patch(
                    Polygon(
                        obj.rectangle(dims),
                        closed=True,
                        alpha=0.7,
                        fill=False,
                        linewidth=0.5,
                        edgecolor="blue",
                    )
                )
            elif obj.type == Geometry.TYPE.POLYTOPE:
                ax.add_patch(
                    Polygon(
                        obj.polygon(dims),
                        closed=True,
                        alpha=0.7,
                        fill=False,
                        linewidth=0.5,
                        edgecolor="red",
                    )
                )
            elif obj.type == Geometry.TYPE.ZONOTOPE:
                g = obj.proj(dims)
                p = Polygon(
                    g.polygon(),
                    closed=True,
                    alpha=1,
                    fill=False,
                    linewidth=0.5,
                    edgecolor="blue",
                )
                ax.add_patch(p)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    for geo in geos:
        __add_patch(geo)

    ax.autoscale_view()
    ax.axis("equal")
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))

    plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


from pyrat.geometry import Geometry
from pyrat.misc import Reachable


def vis2d(r: Reachable.Result, dims: list, width=800, height=800):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    def vis_element(g: Geometry.Base):
        g = g.proj(dims)
        p = plt.Polygon(
            g.polygon(),
            closed=True,
            alpha=1,
            fill=False,
            linewidth=0.5,
            edgecolor="blue",
        )
        ax.add_patch(p)

    if len(r.ti) >= 0:
        for res in r.ti:
            for re in res:
                vis_element(re)
    else:
        for res in r.tp:
            for re in res:
                vis_element(re)

    ax.autoscale_view()

    plt.show()


def vis2dGeo(geos: [Geometry.Base], dims: list, width=800, height=800):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    def __add_patch(obj: Geometry.Base):
        if obj.type == Geometry.TYPE.INTERVAL:
            ax.add_patch(
                Polygon(
                    obj.rectangle(dims),
                    closed=True,
                    alpha=0.7,
                    fill=True,
                    linewidth=0.5,
                    edgecolor="blue",
                )
            )
        else:
            raise NotImplementedError

    for geo in geos:
        __add_patch(geo)

    ax.autoscale_view()

    plt.show()

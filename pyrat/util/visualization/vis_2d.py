from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

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

    plt.axis("equal")
    # ax.axhline(y=0, color="k")
    # ax.axvline(x=0, color="k")
    plt.show()


def vis2dGeo(geos: [Geometry.Base], dims: list, width=800, height=800):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    def __add_interval(obj):
        p = plt.Polygon(
            obj.rectangle(dims),
            closed=True,
            alpha=1,
            fill=False,
            linewidth=0.5,
            edgecolor="blue",
        )
        ax.add_patch(p)

    for geo in geos:
        if geo.type == Geometry.TYPE.INTERVAL:
            __add_interval(geo)
        else:
            raise NotImplementedError

    # final processing
    plt.axis("equal")
    plt.show()

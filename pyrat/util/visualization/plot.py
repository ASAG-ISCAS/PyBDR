import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pyrat.geometry import Geometry, Interval, Zonotope, Polytope
from itertools import chain


def __3d_plot(objs, dims: list, width: int, height: int):
    # TODO
    raise NotImplementedError


def __2d_add_pts(ax, dims, pts: np.ndarray, color):
    assert pts.ndim == 2
    ax.scatter(pts[:, dims[0]], pts[:, dims[1]], c=color)


def __2d_add_interval(ax, i: "Interval", dims, color, filled):
    ax.add_patch(
        Polygon(
            i.proj(dims).rectangle(),
            closed=True,
            alpha=1,
            fill=filled,
            linewidth=1,
            edgecolor=color,
            facecolor=color
        )
    )


def __2d_add_polytope(ax, p: "Polytope", dims, color, filled):
    ax.add_patch(
        Polygon(
            p.polygon(dims),
            closed=True,
            alpha=1,
            fill=filled,
            linewidth=3,
            edgecolor=color,
            facecolor=color
        )
    )


def __2d_add_zonotope(ax, z: "Zonotope", dims, color, filled):
    g = z.proj(dims)
    p = Polygon(
        g.polygon(),
        closed=True,
        alpha=1,
        fill=filled,
        linewidth=1,
        edgecolor=color,
        facecolor=color
    )
    ax.add_patch(p)


def __2d_plot(objs, dims: list, width: int, height: int, xlim=None, ylim=None, c=None, filled=False):
    assert len(dims) == 2
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")

    for i in range(len(objs)):
        this_color = plt.cm.turbo(i / len(objs)) if c is None else c
        geos = (
            [objs[i]]
            if not isinstance(objs[i], list)
            else list(itertools.chain.from_iterable([objs[i]]))
        )
        for geo in geos:
            if isinstance(geo, np.ndarray):
                __2d_add_pts(ax, dims, geo, this_color)
            elif isinstance(geo, Geometry.Base):
                if geo.type == Geometry.TYPE.INTERVAL:
                    __2d_add_interval(ax, geo, dims, "black", filled)
                elif geo.type == Geometry.TYPE.POLYTOPE:
                    __2d_add_polytope(ax, geo, dims, "blue", filled)
                elif geo.type == Geometry.TYPE.ZONOTOPE:
                    __2d_add_zonotope(ax, geo, dims, this_color, filled)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    ax.autoscale_view()
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.show()


def plot(objs, dims: list, mod: str = "2d", width: int = 800, height: int = 800, xlim=None, ylim=None, c=None,
         filled=False):
    if mod == "2d":
        return __2d_plot(objs, dims, width, height, xlim, ylim, c, filled)
    elif mod == "3d":
        return __3d_plot(objs, dims, width, height)
    else:
        raise Exception("unsupported visualization mode")


def __2d_plot_cmp(collections, dims, width, height, xlim, ylim, cs, filled):
    assert len(dims) == 2
    if cs is not None:
        assert len(collections) == len(cs)
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(width * px, height * px), layout='constrained')

    for i in range(len(collections)):
        this_color = plt.cm.turbo(i / len(collections)) if cs is None else cs[i]

        geos = (
            [collections[i]]
            if not isinstance(collections[i], list)
            else list(itertools.chain.from_iterable(collections[i]))
        )

        for geo in geos:
            if isinstance(geo, np.ndarray):
                __2d_add_pts(ax, dims, geo, this_color)
            elif isinstance(geo, Geometry.Base):
                if geo.type == Geometry.TYPE.INTERVAL:
                    __2d_add_interval(ax, geo, dims, "black", filled)
                elif geo.type == Geometry.TYPE.POLYTOPE:
                    __2d_add_polytope(ax, geo, dims, "blue", filled)
                elif geo.type == Geometry.TYPE.ZONOTOPE:
                    __2d_add_zonotope(ax, geo, dims, this_color, filled)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    ax.autoscale_view()
    ax.set_xlabel("x" + str(dims[0]))
    ax.set_ylabel("x" + str(dims[1]))

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.show()


def __3d_plot_cmp(collections, dims, width, height, cs):
    # TODO
    raise NotImplementedError


def plot_cmp(collections, dims: list, mod: str = "2d", width=800, height=800, xlim=None, ylim=None, cs=None,
             filled=False):
    if mod == '2d':
        return __2d_plot_cmp(collections, dims, width, height, xlim, ylim, cs, filled)
    elif mod == '3d':
        return __3d_plot_cmp(collections, dims, width, height, cs)
    else:
        raise Exception("unsupported visulization mode")

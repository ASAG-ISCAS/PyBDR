"""
3D visualization of sets, either interactive with plotly (optional dependency, install with
``pip install pybdr[vis]``) or static with matplotlib.

Every set is projected onto 3 dimensions and drawn as the convex hull of its projected vertices;
all sets are merged into a single mesh, which keeps the rendering fast for thousands of sets.
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.spatial import ConvexHull, QhullError

from pybdr.geometry import Geometry


def _hull(points: np.ndarray) -> ConvexHull:
    """convex hull that also works for flat point sets, by joggling (the returned indices stay exact)"""
    try:
        return ConvexHull(points)
    except QhullError:
        return ConvexHull(points, qhull_options="QJ")


def _zonotope_vertices(c: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """vertices of a zonotope, adding one generator at a time and keeping only the hull vertices"""
    vs = c[None, :]
    for g in gen.T:
        if not np.any(g):
            continue
        vs = np.vstack([vs + g, vs - g])
        if len(vs) > c.size + 1:
            vs = vs[_hull(vs).vertices]
    return vs


def _vertices(geo: Geometry.Base, dims) -> np.ndarray:
    """vertices of the projection of the set onto the given dimensions"""
    dims = list(dims)
    if geo.type == Geometry.TYPE.INTERVAL:
        box = geo.proj(dims)
        return np.array(list(itertools.product(*zip(box.inf, box.sup))), dtype=float)
    if geo.type == Geometry.TYPE.ZONOTOPE:
        if len(dims) == 2:
            # much faster than the general method, which matters for the thousands of sets of a tube
            return geo.proj(dims).polygon()
        return _zonotope_vertices(geo.c[dims], geo.gen[dims, :])
    if geo.type == Geometry.TYPE.POLYTOPE:
        return geo.vertices[:, dims]
    raise NotImplementedError(f"can not plot {geo.type}")


def _hull_mesh(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """points and triangles of the convex hull, flat sets (e.g. zero width) are handled by joggling"""
    return points, _hull(points).simplices


def _flatten(objs) -> tuple[list, np.ndarray]:
    """sets and the index of their group, nested lists (e.g. per time step) form one group each"""
    geos, groups = [], []
    for i, obj in enumerate(objs):
        members = obj if isinstance(obj, (list, tuple)) else [obj]
        geos.extend(members)
        groups.extend([i] * len(members))
    return geos, np.asarray(groups)


def _merge(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """concatenate meshes, returns points, triangles and the mesh index of every triangle"""
    offsets = np.cumsum([0] + [len(p) for p, _ in meshes[:-1]])
    points = np.concatenate([p for p, _ in meshes])
    triangles = np.concatenate([t + o for (_, t), o in zip(meshes, offsets)])
    owner = np.concatenate([np.full(len(t), k) for k, (_, t) in enumerate(meshes)])
    return points, triangles, owner


def _face_colors(values: np.ndarray, color, colormap: str) -> np.ndarray:
    """RGBA color of every face, by the given color or by the values mapped through the colormap"""
    if color is not None:
        return np.tile(np.asarray(to_rgba(color)), (len(values), 1))
    span = max(values.max() - values.min(), 1)
    return plt.get_cmap(colormap)((values - values.min()) / span)


def _rgb(rgba) -> str:
    return f"rgb({rgba[0] * 255:.0f},{rgba[1] * 255:.0f},{rgba[2] * 255:.0f})"


def _render(points, triangles, face_values, labels, backend, color, colormap, opacity,
            width, height, show, save_file_name, offline):
    if backend == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError as err:
            raise ImportError("the plotly backend needs plotly, install it with: pip install pybdr[vis]") from err

        if color is not None:
            coloring = dict(color=_rgb(to_rgba(color)))
        else:
            # one value per face mapped through the colormap, much smaller than one color string per face
            cmap = plt.get_cmap(colormap)
            coloring = dict(intensity=face_values, intensitymode="cell", showscale=False,
                            colorscale=[[v, _rgb(cmap(v))] for v in np.linspace(0, 1, 11)])
        fig = go.Figure(go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            opacity=opacity, flatshading=True, hoverinfo="skip", **coloring,
        ))
        fig.update_layout(
            width=width, height=height, margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2]),
        )
        if save_file_name is not None:
            fig.write_html(save_file_name, include_plotlyjs=True if offline else "cdn")
        if show:
            fig.show()
        return fig

    if backend == "matplotlib":
        colors = _face_colors(face_values, color, colormap)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        px = 1 / plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width * px, height * px), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        colors[:, 3] = opacity
        ax.add_collection3d(Poly3DCollection(points[triangles], facecolors=colors, edgecolors="none"))
        lo, hi = points.min(axis=0), points.max(axis=0)
        pad = 0.05 * np.maximum(hi - lo, 1e-9)
        ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(lo[1] - pad[1], hi[1] + pad[1]),
               zlim=(lo[2] - pad[2], hi[2] + pad[2]), xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
        if save_file_name is not None:
            fig.savefig(save_file_name)
        if show:
            plt.show()
        return fig, ax

    raise ValueError(f"unknown backend {backend!r}, use 'plotly' or 'matplotlib'")


def plot3d(objs, dims, backend: str = "plotly", color=None, colormap: str = "turbo", opacity: float = 0.5,
           width: int = 800, height: int = 800, show: bool = True, save_file_name=None, offline: bool = False):
    """
    plot sets projected onto 3 dimensions

    :param objs: list of Interval / Zonotope / Polytope, or list of lists of them (e.g. the reachable
        sets of several cells per time step), every entry of the outer list gets its own color
    :param dims: the 3 dimensions to plot
    :param backend: "plotly" (interactive, needs ``pip install pybdr[vis]``) or "matplotlib" (static)
    :param color: one color for all sets, by default the sets are colored by their order with the colormap
    :param colormap: matplotlib colormap name used when no color is given
    :param opacity: opacity of the sets
    :param show: show the figure
    :param save_file_name: save the figure, as html for plotly and as image for matplotlib
    :param offline: plotly only, embed plotly.js in the html file (+4 MB) so it opens without internet
    :return: plotly Figure, or matplotlib figure and axes
    """
    assert len(dims) == 3
    geos, groups = _flatten(objs)
    points, triangles, owner = _merge([_hull_mesh(_vertices(geo, dims)) for geo in geos])
    labels = [f"x{d}" for d in dims]
    return _render(points, triangles, groups[owner], labels, backend, color, colormap, opacity,
                   width, height, show, save_file_name, offline)


def plot_tube(objs, dims, step: float, t_start: float = 0.0, backend: str = "plotly", color=None,
              colormap: str = "turbo", opacity: float = 0.5, width: int = 800, height: int = 800,
              show: bool = True, save_file_name=None, offline: bool = False):
    """
    plot the evolution of sets over time as a tube: the projection onto 2 dimensions of the k-th
    entry of objs is extruded along the time axis over [t_start + k * step, t_start + (k + 1) * step]

    :param objs: time ordered list of sets, or of lists of sets (e.g. several cells per time step)
    :param dims: the 2 state dimensions to plot
    :param step: time step between consecutive entries of objs
    :param t_start: time of the first entry
    :param backend: see plot3d, as well as color, colormap, opacity, width, height, show, save_file_name, offline
    :return: plotly Figure, or matplotlib figure and axes
    """
    assert len(dims) == 2
    geos, groups = _flatten(objs)
    meshes = []
    for geo, k in zip(geos, groups):
        polygon = _vertices(geo, dims)
        t0, t1 = t_start + k * step, t_start + (k + 1) * step
        prism = np.vstack([np.column_stack([polygon, np.full(len(polygon), t)]) for t in (t0, t1)])
        meshes.append(_hull_mesh(prism))
    points, triangles, owner = _merge(meshes)
    labels = [f"x{dims[0]}", f"x{dims[1]}", "t"]
    return _render(points, triangles, groups[owner], labels, backend, color, colormap, opacity,
                   width, height, show, save_file_name, offline)

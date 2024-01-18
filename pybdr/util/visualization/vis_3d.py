import numpy as np

# import open3d as o3d


from pybdr.geometry import Geometry


def __vis_line_set(pts, line_set):
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(line_set),
    )
    colors = np.zeros((pts.shape[0], 3), dtype=float)
    colors[:, 2] = 1
    lines.colors = o3d.utility.Vector3dVector(colors)
    return lines


def vis3d():
    raise NotImplementedError
    # TODO


def vis3dGeo(
    geos: [Geometry.Base],
    dims: list,
    width=1920,
    height=1080,
    window_name="PyRAT",
):
    vis_geos = []
    for geo in geos:
        if geo.type == Geometry.TYPE.INTERVAL:
            pts, lines = geo.cube(dims)
            vis_geos.append(__vis_line_set(pts, lines))
    o3d.visualization.draw(
        vis_geos,
        title=window_name,
        width=width,
        height=height,
        line_width=3,
        bg_color=(52 / 255, 52 / 255, 52 / 255, 1.0),
        show_skybox=False,
        show_ui=True,
    )

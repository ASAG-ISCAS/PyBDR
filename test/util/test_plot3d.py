import numpy as np
import pytest

from pybdr.geometry import Interval, Polytope, Zonotope
from pybdr.geometry.polytope import vertices_to_halfspaces
from pybdr.util.visualization import plot3d, plot_tube
from pybdr.util.visualization.plot3d import _vertices, _zonotope_vertices

RNG = np.random.default_rng(0)
CUBE = Interval([0, 0, 0], [1, 2, 3])
ZONO = Zonotope([0, 0, 0, 0], RNG.uniform(-1, 1, (4, 6)))
POLY = Polytope(*vertices_to_halfspaces(RNG.uniform(-1, 1, (20, 3))))


def test_interval_vertices():
    vs = _vertices(CUBE, [0, 1, 2])
    assert vs.shape == (8, 3)
    assert np.array_equal(vs.min(axis=0), CUBE.inf) and np.array_equal(vs.max(axis=0), CUBE.sup)


def test_zonotope_vertices_are_extreme_points():
    vs = _vertices(ZONO, [0, 2, 3])
    # every vertex is attained by a sign combination of the generators
    gen = ZONO.gen[[0, 2, 3]]
    signs = np.array(np.meshgrid(*[[-1, 1]] * gen.shape[1])).reshape(gen.shape[1], -1)
    corners = (gen @ signs).T
    assert all(np.min(np.linalg.norm(corners - v, axis=1)) < 1e-9 for v in vs)


def test_flat_zonotope():
    # rank 2 generators in 3D, the hull is computed by joggling
    vs = _zonotope_vertices(np.zeros(3), np.array([[1.0, 0, 1], [0, 1, 1], [0, 0, 0]]))
    assert np.allclose(vs[:, 2], 0)
    # joggling may keep points inside the flat hull, which does not change the drawn shape
    from scipy.spatial import ConvexHull

    assert len(ConvexHull(vs[:, :2]).vertices) == 6


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_plot3d(backend, tmp_path):
    if backend == "plotly":
        pytest.importorskip("plotly")
    out = tmp_path / ("fig.html" if backend == "plotly" else "fig.png")
    # nested lists: one color per entry of the outer list
    result = plot3d([CUBE, [ZONO.proj([0, 1, 2]), POLY]], [0, 1, 2], backend=backend, show=False, save_file_name=out)
    assert out.stat().st_size > 0
    if backend == "plotly":
        mesh = result.data[0]
        assert len(result.data) == 1 and len(mesh.i) == len(mesh.intensity)
        assert len(set(mesh.intensity)) == 2


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_plot_tube(backend):
    if backend == "plotly":
        pytest.importorskip("plotly")
    sets = [Zonotope([k * 0.1, 0], np.eye(2) * 0.2) for k in range(5)]
    result = plot_tube(sets, [0, 1], step=0.5, backend=backend, show=False)
    if backend == "plotly":
        z = result.data[0].z
        assert np.isclose(min(z), 0) and np.isclose(max(z), 2.5)


def test_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        plot3d([CUBE], [0, 1, 2], backend="open3d", show=False)

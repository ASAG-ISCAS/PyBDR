import numpy as np
import pytest

from pybdr.geometry import Zonotope
from pybdr.util.visualization import plot, plot_cmp

# a tall and narrow set of zonotopes
SETS = [Zonotope([0.3 * np.sin(t), 4 * np.cos(t / 2)], np.diag([0.05, 0.1])) for t in np.linspace(0, 6, 40)]


@pytest.mark.parametrize("plot_fn", [lambda **kw: plot(SETS, [0, 1], **kw), lambda **kw: plot_cmp([SETS], [0, 1], **kw)])
def test_auto_aspect_fits_the_data(plot_fn):
    _, ax = plot_fn(show=False)
    # the x axis spans the data (about 0.7 wide), not the 8 units of the y axis
    assert np.diff(ax.get_xlim())[0] < 1.0
    assert np.diff(ax.get_ylim())[0] > 8.0


def test_equal_aspect():
    fig, ax = plot(SETS, [0, 1], show=False, aspect="equal")
    fig.canvas.draw()
    assert np.isclose(np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0], rtol=0.05)


def test_unknown_aspect():
    with pytest.raises(ValueError, match="aspect"):
        plot(SETS, [0, 1], show=False, aspect="square")

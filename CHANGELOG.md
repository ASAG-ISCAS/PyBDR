# Changelog

## 1.1.0

Packaging: PyBDR installs with pip or conda without any system library, on Linux, macOS and Windows.

### Breaking changes

- Python 3.11 or newer is required.
- RealPaver was removed (`pybdr.util.functional.RealPaver`). The boundary of polytopes and zonotopes
  is computed with codac; `boundary(src, r, elem)` keeps its interface.
- pypoman was removed. Polytope operations use scipy; degenerate polytopes (empty, unbounded,
  empty interior) raise `ValueError`.
- The unused open3d based `pybdr.util.gui` module and the empty `misc`, `discrete_system` and
  `hybrid_system` packages were removed.
- codac 2.1 (release) is required instead of the 2.0 pre-releases.

### Added

- `plot3d` and `plot_tube`: 3D visualization with plotly (interactive, `pip install "pybdr[vis]"`)
  or matplotlib.
- `polytope_boundary` and `function_boundary` in `pybdr.util.functional`.
- Plot functions accept `show` and `save_file_name` and return the figure; 2D plots adapt the axes to
  the data by default (`aspect="equal"` keeps the same scale on both axes, the previous behavior).
- Conda recipes (`conda-recipe/`), soundness and regression tests, CI.
- Example notebooks for every algorithm (replacing the demo scripts in `test/algorithm/`) and a Colab
  demo, stored with their outputs; `scripts/run_checks.py` runs the tests and the notebooks.

### Fixed

- Interval matrix products modified the given matrix / the bounds of the operands in place.
- Interval evaluation of the models failed with numpy >= 2.5.
- `extract_boundary` crashed for polytopes with 12 or more constraints.
- `Simulator` did not work for linear systems.
- `reach_parallel` failed on macOS and Windows for dynamics defined in `__main__` (e.g. in a notebook)
  or in a function.
- `XSE2016CAV` needed the GLPK solver, which fresh installs do not have.
- `Polytope.proj` failed for projections onto other than 2 dimensions.
- `plot_cmp` saved an empty image when the figure was also shown.
- `sympy` was missing from the dependencies; the license metadata said MIT instead of GPLv3.

# Conda packages

codac is not on conda-forge yet, so three packages are built:

| recipe  | package                                                   |
|---------|-----------------------------------------------------------|
| `vibes` | dependency declared by codac, pure Python (`noarch`)      |
| `codac` | the official PyPI wheels repackaged, one per Python version |
| `pybdr` | PyBDR itself, pure Python (`noarch`)                      |

Build them into a local channel with [rattler-build](https://rattler.build)
(included in `environment-dev.yml`), in this order:

```bash
rattler-build build --recipe conda-recipe/vibes --output-dir ./conda-channel -c conda-forge
rattler-build build --recipe conda-recipe/codac --variant-config conda-recipe/variants.yaml \
    --output-dir ./conda-channel -c conda-forge
rattler-build build --recipe conda-recipe/pybdr --output-dir ./conda-channel -c conda-forge
```

The pybdr build runs the test suite (`pytest -m "not slow"`). Install from the local channel with:

```bash
conda create -n pybdr -c file://$PWD/conda-channel -c conda-forge pybdr
```

Notes:

- The codac package is platform specific, build it on every platform it is needed for
  (it downloads the matching wheel from PyPI).
- Keep the version in `pybdr/recipe.yaml` in sync with `pybdr/__init__.py`.

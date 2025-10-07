# Installation Guide for PyBDR

PyBDR is a set-boundary based reachability analysis toolbox for Python.

## Prerequisites

- **Conda or Miniconda** ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- Git (for cloning the repository)

## Quick Installation (3 Steps)

```bash
# Step 1: Create conda environment with all dependencies
conda env create -f environment.yml

# Step 2: Activate the environment
conda activate pybdr

# Step 3: Install PyBDR
pip install -e .
```

**That's it!** The `environment.yml` automatically installs Python 3.8+, all dependencies, and the required `cddlib` C library.

## Alternative: Install in Existing Conda Environment

If you already have a conda environment:

```bash
# Activate your environment
conda activate your_env_name

# Install cddlib C library
conda install -c conda-forge cddlib

# Install PyBDR (this will install all Python dependencies including codac pre-release)
pip install -e .
```

## For Developers

```bash
# Clone and setup
git clone https://github.com/ASAG-ISCAS/PyBDR.git
cd PyBDR
conda env create -f environment.yml
conda activate pybdr

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Verifying Installation

After installation, verify that PyBDR is correctly installed:

```python
import pybdr
from pybdr.geometry import Interval, Zonotope
from pybdr.algorithm import ASB2008CDC

# Create a simple interval
interval = Interval([0, 1], [1, 2])
print(f"Interval created: {interval}")

print("PyBDR successfully installed!")
```

## Troubleshooting

### Issue: `pycddlib` build fails with "cddlib/setoper.h not found"

**Cause**: The `cddlib` C library is not installed.

**Solution**: Make sure you've installed `cddlib` via conda:

```bash
conda install -c conda-forge cddlib
```

### Issue: CVXPY installation fails

**Solution**: CVXPY is included in `environment.yml`. If you're installing manually, use conda:

```bash
conda install -c conda-forge cvxpy
```

### Issue: "No module named 'pybdr'" after installation

**Solution**: Make sure you've activated the correct conda environment:

```bash
conda activate pybdr
```

### Issue: Codac installation or "No module named 'codac'"

**Cause**: Codac v2 is only available as a pre-release version.

**Solution**: Codac pre-release is automatically included when you run `pip install -e .`. If you need to install it manually:

```bash
pip install "codac>=2.0.0.dev20"
```

Or reinstall PyBDR:

```bash
pip install -e . --force-reinstall
```

## Package Structure

After installation, PyBDR provides the following modules:

```
pybdr/
├── algorithm/          # Reachability analysis algorithms
├── dynamic_system/     # System models (continuous, discrete, hybrid)
├── geometry/           # Geometric representations (intervals, zonotopes, polytopes)
├── model/              # Pre-defined system models
├── util/               # Utility functions
└── misc/               # Miscellaneous utilities
```

## Updating PyBDR

If you installed in development mode (`pip install -e .`), simply pull the latest changes:

```bash
cd PyBDR
git pull origin master
```

## Uninstalling

To remove PyBDR:

```bash
pip uninstall pybdr
```

To remove the entire conda environment:

```bash
conda deactivate
conda env remove -n pybdr
```

## Support

For issues and questions:

- GitHub Issues: https://github.com/ASAG-ISCAS/PyBDR/issues
- Documentation: https://asag-iscas.github.io/docs.pybdr/

## Dependencies

PyBDR depends on:

- **numpy** (≥1.20.0): Numerical computations
- **scipy** (≥1.7.0): Scientific computing
- **pypoman** (≥0.5.4): Polytope manipulation
- **cvxpy** (≥1.1.0): Convex optimization
- **matplotlib** (≥3.3.0): Visualization

**System dependencies (automatically installed via conda):**

- **cddlib**: C library for polytope computations (required by pypoman)

All dependencies are automatically installed when using the `environment.yml` file with conda.

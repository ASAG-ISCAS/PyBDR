# Examples

The notebooks are stored with their outputs, so the results can be viewed here on GitHub without
running anything. Each one also opens in Colab, where its first cell installs PyBDR.

| notebook | content | | runtime |
|---|---|---|---|
| [colab_demo](colab_demo.ipynb) | collision verification: can a car changing lanes from a non-convex set of initial states hit an obstacle? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASAG-ISCAS/PyBDR/blob/master/examples/colab_demo.ipynb) | 2 min |
| [asb2008cdc](asb2008cdc.ipynb) | nonlinear systems, conservative linearization (14 cases) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASAG-ISCAS/PyBDR/blob/master/examples/asb2008cdc.ipynb) | 11 min |
| [alth2013hscc](alth2013hscc.ipynb) | nonlinear systems, conservative polynomialization (5 cases) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASAG-ISCAS/PyBDR/blob/master/examples/alth2013hscc.ipynb) | 4 min |
| [gira2005hscc](gira2005hscc.ipynb) | linear systems with zonotopes (7 cases) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASAG-ISCAS/PyBDR/blob/master/examples/gira2005hscc.ipynb) | 12 min |
| [alk2011hscc](alk2011hscc.ipynb) | linear systems with uncertain inputs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASAG-ISCAS/PyBDR/blob/master/examples/alk2011hscc.ipynb) | 30 s |
| [xse2016cav](xse2016cav.ipynb) | under-approximation of backward reachable sets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASAG-ISCAS/PyBDR/blob/master/examples/xse2016cav.ipynb) | 30 s |

The runtimes were measured on an Apple Silicon Mac; Colab (2 CPUs) is slower.
The interactive plotly figure at the end of `colab_demo` only shows when the notebook is run (GitHub
does not display it); the static figure above it shows the same.

## Before a commit

```bash
python scripts/run_checks.py
```

runs the tests and all notebooks (about 30 minutes), saves the outputs of the notebooks that run
through and writes `reports/summary.md`. Notebooks that fail are saved to `reports/failed/` with the
error, the stored version stays unchanged. Pass notebook paths to run only those, or
`--skip-notebooks` for the tests only.

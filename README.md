<p align="center">
Set-boundary based Reachability Analysis Toolbox in Python
</p>

<p align="center">
    <br />
        <a href="https://asag-iscas.github.io/docs.pybdr/"><strong>Online Documents »</strong></a>
    <br />    
</p>

# Motivation

Reachability analysis, which computes sets of states reachable by a system over time, plays a fundamental role in the temporal verification of nonlinear systems. In practice, however, overly pessimistic over-approximations often render many temporal properties unverifiable. This pessimism mainly arises from the wrapping effect, namely the propagation and accumulation of over-approximation errors during the iterative construction of reachable sets.
Since the severity of the wrapping effect strongly correlates with the volume of the initial set, partitioning-based techniques—where the initial state space is divided into smaller subsets and analyzed independently—are commonly employed to mitigate this effect, especially for large initial sets and long time horizons
(<a href="https://ieeexplore.ieee.org/document/7585104"><strong>see here</strong></a>).
Such partitioning, however, typically incurs substantial computational and memory overhead, often making existing reachability analysis techniques unsuitable for complex real-world applications. In particular, being forced to explore the full—often exponential in the system dimension—number of partitions severely limits scalability.
Motivated by this challenge, this tool implements the so-called
<a href="http://lcs.ios.ac.cn/~xuebai/publication.html"><strong>set-boundary–based method</strong></a>,
which computes the full reachable state space by performing state-exploratory analysis on only a small sub-volume of the initial set, namely a set enclosing its boundary. By avoiding exhaustive exploration of the interior, this approach significantly improves scalability while preserving soundness.
For theoretical foundations, please refer to
<a href="https://ieeexplore.ieee.org/document/7585104"><strong>Bai Xue et al., “Reach-Avoid Verification for Nonlinear Systems Based on Boundary Analysis,” IEEE Transactions on Automatic Control, 2017</strong></a>, and
<a href="https://ieeexplore.ieee.org/document/9023360"><strong>Bai Xue et al., “Over- and Under-Approximating Reach Sets for Perturbed Delay Differential Equations,” IEEE Transactions on Automatic Control, 2020</strong></a>.

The set-boundary–based method can be used to perform reachability analysis for systems modeled by:

1. Ordinary differential equations (ODEs) with Lipschitz-continuous perturbations,
2. Delay differential equations (DDEs) with Lipschitz-continuous perturbations,
3. Neural ordinary differential equations (Neural ODEs).

# Installation

PyBDR requires Python 3.11 or newer and runs on Linux, macOS and Windows. All dependencies are
regular Python packages, no system library has to be installed.

## With pip

```bash
pip install "pybdr @ git+https://github.com/ASAG-ISCAS/PyBDR"
```

Optional extras:

- `pybdr[vis]` adds [plotly](https://plotly.com/python/) for interactive 3D plots.
- `pybdr[test]` adds pytest to run the test suite.

## With conda

codac, the interval analysis library PyBDR uses, is not on conda-forge yet. The recipes in
[conda-recipe/](conda-recipe/README.md) build conda packages of PyBDR and codac into a local channel,
which can then be installed with:

```bash
conda create -n pybdr -c file://$PWD/conda-channel -c conda-forge pybdr
```

## From source

To work on PyBDR, create the conda environment, which installs PyBDR in editable mode:

```bash
git clone https://github.com/ASAG-ISCAS/PyBDR.git
cd PyBDR
conda env create -f environment-dev.yml   # environment.yml without the test and packaging tools
conda activate pybdr-dev
pytest -m "not slow"                      # the slow tests execute the example notebooks
```

or, without conda, `pip install -e ".[dev]"`. Before a commit, `python scripts/run_checks.py` runs the
tests and all example notebooks and writes a report to `reports/summary.md`
(see [examples/](examples/README.md)).

## Google Colab

Open the [demo notebook](examples/colab_demo.ipynb) in Colab with the button below, its first cell
installs PyBDR. It verifies whether a car changing lanes can hit obstacles, starting from a non-convex
set of initial states. [examples/](examples/README.md) has a notebook for every algorithm, stored with its
results so that they can be viewed on GitHub directly.

# How to use [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASAG-ISCAS/PyBDR/blob/master/examples/colab_demo.ipynb)

## Computing Reachable Sets based on Boundary Analysis for Nonlinear Systems

The tool provides sample files which serve as demonstrations of the proper utilization for computing reachable sets. These sample files serve as a reference point for users to grasp the process of modifying the dynamics and parameters necessary for reachability analysis. This feature aids users in experimenting with their analyses, allowing them to assess the impact of different settings on the overall computation of the reachable sets

<!--The tool comes with sample files that demonstrate how it should be utilized to compute reachable sets. By referring to
these sample files, users can gain an understanding of how to modify the dynamics and parameters required for
reachability analysis. This feature helps users experiment with their analysis by using different settings to assess
their effects on the overall computation of the reachable sets.-->

For example, consider the following dynamic system:

$$
\begin{align*}
\dot{x} &= y \\
\dot{y} &= (1-x^2)y-x
\end{align*}
$$

```python
import numpy as np
from pybdr.algorithm import ASB2008CDC
from pybdr.util.functional import performance_counter, performance_counter_start
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.geometry.operation import boundary, cvt2
from pybdr.model import *
from pybdr.util.visualization import plot

# reach_parallel starts worker processes, which requires this guard on macOS and Windows
if __name__ == "__main__":
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 6.74
    options.step = 0.005
    options.tensor_order = 3
    options.taylor_terms = 4
    options.u = Zonotope.zero(1, 1)
    options.u_trans = np.zeros(1)

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([1.23, 2.34], [1.57, 2.46])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 1, Geometry.TYPE.ZONOTOPE)

    this_time = performance_counter_start()
    ri_without_bound, rp_without_bound = ASB2008CDC.reach(vanderpol, [2, 1], options, x0)
    this_time = performance_counter(this_time, 'reach_without_bound')

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(vanderpol, [2, 1], options, xs)
    this_time = performance_counter(this_time, 'reach_with_bound')

    # visualize the results
    plot(ri_without_bound, [0, 1])
    plot(ri_with_bound, [0, 1])
```

|     With Boundary Analysis (BA)     |       No Boundary Analysis (NBA)       |
| :---------------------------------: | :------------------------------------: |
| ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/vanderpol_bound.png) | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/vanderpol_no_bound.png) |

For large initial sets,

|                                                         System                                                          |                                                                  Code                                                                   |   Reachable Sets (Orange-NBA,Blue-BA)   |
| :---------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------: |
|        [synchronous machine](https://github.com/ASAG-ISCAS/PyBDR/blob/master/pybdr/model/synchronous_machine.py)        | [benchmark_synchronous_machine_cmp.py](https://github.com/ASAG-ISCAS/PyBDR/blob/master/benchmarks/benchmark_synchronous_machine_cmp.py) |   ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/sync_machine_cmp.png)    |
| [Lotka Volterra model of 2 variables](https://github.com/ASAG-ISCAS/PyBDR/blob/master/pybdr/model/lotka_volterra_2d.py) |   [benchmark_lotka_volterra_2d_cmp.py](https://github.com/ASAG-ISCAS/PyBDR/blob/master/benchmarks/benchmark_lotka_volterra_2d_cmp.py)   | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/lotka_volterra_2d_cmp.png) |
|                 [Jet engine](https://github.com/ASAG-ISCAS/PyBDR/blob/master/pybdr/model/jet_engine.py)                 |          [benchmark_jet_engine_cmp.py](https://github.com/ASAG-ISCAS/PyBDR/blob/master/benchmarks/benchmark_jet_engine_cmp.py)          |    ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/jet_engine_cmp.png)     |

For large time horizons, i.e. consider
the system [Brusselator](https://github.com/ASAG-ISCAS/PyBDR/blob/master/pybdr/model/brusselator.py)

> For more details about the following example, please refer to
> our [code](https://github.com/ASAG-ISCAS/PyBDR/blob/master/benchmarks/benchmark_brusselator_cmp.py).

| Time instance | With Boundary Analysis                |        Without Boundary Analysi        |
| :-----------: | ------------------------------------- | :------------------------------------: |
|     t=5.4     | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/brusselator_ba_t5.4.png) | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/brusselator_nba_t5.4.png) |
|     t=5.7     | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/brusselator_ba_t5.7.png) | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/brusselator_nba_t5.7.png) |
|     t=6.0     | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/brusselator_ba_t6.png)   |  ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/brusselator_nba_t6.png)  |
|     t=6.1     | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/brusselator_ba_t6.1.png) |      **Set Explosion Occurred!**       |

## Computing Reachable Sets based on Boundary Analysis for Neural ODE

For example, consider a neural ODE with the following parameters and $\textit{sigmoid}$ activation function, also
evaluated
in <a href="https://link.springer.com/content/pdf/10.1007/978-3-031-15839-1_15.pdf"><strong>'Manzanas Lopez, D., Musau,
P., Hamilton, N. P., & Johnson, T. T. Reachability analysis of a general class of neural ordinary differential
equations. In Formal Modeling and Analysis of Timed Systems: 20th International Conference, FORMATS 2022, Warsaw,
Poland, September 13–15, 2022, Proceedings (pp. 258-277).'</strong></a>:

$$
w_1 = \left[
\begin{align*}
0.2911133 \quad & 0.12008807\\
-0.24582624 \quad & 0.23181419\\
-0.25797904 \quad & 0.21687193\\
-0.19282854 \quad & -0.2602416 \\
0.26780415 \quad & -0.20697702\\
0.23462369\quad & 0.2294843 \\
-0.2583547\quad & 0.21444395\\
-0.04514714 \quad & 0.29514763\\
-0.15318371 \quad & -0.275755 \\
0.24873598 \quad & 0.21018365
\end{align*}
\right]
$$

$$
w_2 = \left[
\begin{align*}
-0.58693904 \quad & -0.814841 & -0.8175157 \quad & 0.97060364 & 0.6908913\\
-0.92446184 \quad & -0.79249185 & -1.1507587 \quad & 1.2072723 & -0.7983982\\
1.1564877 \quad & -0.8991244 & -1.0774536 \quad & -0.6731967 & 1.0154784\\
0.8984464 \quad & -1.0766245 & -0.238209 \quad & -0.5233613 & 0.8886671
\end{align*}
\right]
$$

$$
b_1 = \left[
\begin{align*}
0.0038677\quad & -0.00026365 & -0.007168970\quad & 0.02469357 & 0.01338706\\
0.00856025\quad & -0.00888401& 0.00516089\quad & -0.00634514 & -0.01914518
\end{align*}
\right]
$$

$$
b_2 = \left[
\begin{align*}
-0.04129209 \quad & -0.01508532
\end{align*}
\right]
$$

```python
import numpy as np

from pybdr.algorithm import ASB2008CDC
from pybdr.geometry import Zonotope, Interval, Geometry
from pybdr.model import *
from pybdr.util.visualization import plot, plot_cmp
from pybdr.geometry.operation import boundary, cvt2
from pybdr.util.functional import performance_counter_start, performance_counter

# reach_parallel starts worker processes, which requires this guard on macOS and Windows
if __name__ == "__main__":
    # settings for the computation
    options = ASB2008CDC.Options()
    options.t_end = 1
    options.step = 0.01
    options.tensor_order = 2
    options.taylor_terms = 2

    options.u = Zonotope([0], np.diag([0]))
    options.u_trans = options.u.c

    # settings for the using geometry
    Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
    Zonotope.ORDER = 50

    z = Interval([0, -0.5], [1, 0.5])
    x0 = cvt2(z, Geometry.TYPE.ZONOTOPE)
    xs = boundary(z, 2, Geometry.TYPE.ZONOTOPE)

    print(len(xs))

    this_time = performance_counter_start()
    ri_without_bound, rp_without_bound = ASB2008CDC.reach(neural_ode_spiral1, [2, 1], options, x0)
    this_time = performance_counter(this_time, "reach_without_bound")

    ri_with_bound, rp_with_bound = ASB2008CDC.reach_parallel(neural_ode_spiral1, [2, 1], options, xs)
    this_time = performance_counter(this_time, "reach_with_bound")

    # visualize the results
    plot_cmp([ri_without_bound, ri_with_bound], [0, 1], cs=["#FF5722", "#303F9F"])
```

In the following table, we show the reachable computed with boundary analysis and without boundary analysis on different
time instance cases.

| Time Instance |    With Boundary Analysis     |  Without Boundary Analysis   |
| :-----------: | :---------------------------: | :--------------------------: |
|     t=0.5     | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/Neural_BA05.png) | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/Neural_E05.png) |
|     t=1.0     | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/Neural_BA1.png)  | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/Neural_E1.png)  |
|     t=1.5     | ![](https://raw.githubusercontent.com/ASAG-ISCAS/PyBDR/master/doc/imgs/Neural_BA15.png) |  **Set Explosion Occured!**  |

## 3D Visualization

Besides the 2D `plot` and `plot_cmp`, sets can be shown in 3D with `plot3d` (projection onto 3 state
dimensions) and `plot_tube` (2 state dimensions over time). The interactive plotly backend works in
Jupyter and Colab and needs `pip install "pybdr[vis]"`; `backend="matplotlib"` draws static figures.

```python
from pybdr.geometry import Geometry, Interval
from pybdr.geometry.operation import boundary
from pybdr.util.visualization import plot3d, plot_tube

# reachable sets of the example above over time
plot_tube(ri_without_bound[1:], [0, 1], step=options.step)

# boxes covering the boundary of a cube, as a static figure
cells = boundary(Interval([0, 0, 0], [1, 1, 1]), 0.25, Geometry.TYPE.INTERVAL)
plot3d(cells, [0, 1, 2], backend="matplotlib", save_file_name="cells.png", show=False)
```

All plot functions accept `show=False` and `save_file_name` and return the figure for further changes.

## Frequently Asked Questions and Troubleshooting

### The computation is slow

Two modes of computation are supported by the tool for reachable sets. One mode is to compute the reachable set of evolved states using the entire initial set in a set propagation manner, while the other mode is to compute the reachable set of evolved states based on the boundary of the initial state set.

The computation may be slow for several reasons such as large computational time intervals, small steps, high Taylor expansion orders, or a large number of state variables.

To accelerate the computations, experiments can be performed with a smaller computational time horizon, a smaller order of expansion (such as 2), and a larger time step. Then gradually increase the computational time horizon and order of expansion based on the results of this setting to achieve the desired set of reachable states at an acceptable time consumption.

<!--Two modes of computation are supported by the tool for reachable sets. One mode is to compute the reachable set of
evolved states using the entire initial set in a set propagation manner, while the other mode is to compute the
reachable set of evolved states based on the boundary of the initial state set.

The computation may be slow for several reasons such as large computational time intervals, small steps, high Taylor
expansion orders, or a large number of state variables.

To accelerate the computations, experiments can be performed with a smaller computational time horizon, a smaller
order
of expansion (such as 2), and a larger time step. Then gradually increase the computational time horizon and order of
expansion based on the results of this setting to achieve the desired set of reachable states at an acceptable time
consumption.-->

### `reach_parallel` fails on macOS or Windows

`reach_parallel` computes the cells of the boundary in worker processes. On macOS and Windows these
processes import the main script again, so the script must put its computations under
`if __name__ == "__main__":` as in the examples above. Notebooks (Jupyter, Colab) need no guard, and
dynamics defined in a notebook work with `reach_parallel` as well.

### Controlling the wrapping effect

To enhance the precision of the reachable set computation, one can split the boundaries of initial sets or increase the
order of the Taylor expansion while reducing the step size.

> Feel free to contact [dingjianqiang0x@gmail.com](mailto:dingjianqiang0x@gmail.com) if you find any
> issues or bugs in this code, or you struggle to run it in any way.

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE.md) file for
details.

## Citing PyBDR

```
@inproceedings{ding2024pybdr,
  title={PyBDR: Set-Boundary Based Reachability Analysis Toolkit in Python},
  author={Ding, Jianqiang and Wu, Taoran and Liang, Zhen and Xue, Bai},
  booktitle={International Symposium on Formal Methods},
  pages={140--157},
  year={2024},
  organization={Springer}
}
```

## Acknowledgement

When developing this tool, we drew upon models used in other tools for calculating reachable sets, including Flow\*, CORA, and various others.

<!--When creating this tool, reference was made to models utilized in other reachable set calculation tools such as Flow*,
CORA, and others.-->

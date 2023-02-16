<p align="center">
The Set-boundary based Reachability Analysis Toolbox in Python
</p>


<p align="center">
    <br />
        <a href="https://asag-iscas.github.io/docs.pybdr/"><strong>Online Documents »</strong></a>
    <br />    
</p>

# Motivation

Reachability analysis, which involves computing reachable state sets, plays a fundamental role in the temporal
verification of nonlinear systems. Overly pessimistic over-approximations, however, render many temporal properties
unverifiable in practice. This pessimism mainly arises due to the wrapping effect, which is the propagation and
accumulation of over-approximation error through the iterative computation in the construction of reachable sets. As the
extent of the wrapping effect correlates strongly with the volume of the initial set, techniques that partition the
initial state space and independently compute reachable sets of those partitions are often used to reduce the wrapping
effect, especially for large initial sets or/and large time horizons. Such partitioning may, however, induce extensive
demand on computation time and memory, often rendering the existing reachability analysis techniques not suitable for
complex real-world applications. Not being forced to explore the full, i.g. exponential in the dimensionality, number of
partitions could help such procedures tremendously. This is the theme of this tool, which implements the
so-called <a href="http://lcs.ios.ac.cn/~xuebai/publication.html"><strong>'set-boundary based method'</strong></a> that
explores means of computing
the full reachable state space based on state-exploratory analysis of just a small sub-volume of the initial state set,
namely a set enclosing its boundary. For theoretical analysis, please refer to
<a href="https://ieeexplore.ieee.org/document/7585104"><strong>'Bai Xue, Arvind Easwaran, Nam-Joon Cho and Martin
Fränzle.Reach-Avoid Verification for Nonlinear Systems Based on Boundary Analysis. IEEE Transactions on Automatic
Control (IEEE TAC), vol. 62: 3518--3523, 2017.'</strong></a> and
<a href="https://ieeexplore.ieee.org/document/7585104"><strong>'Bai Xue, Qiuye Wang, Shenghua Feng, and Naijun Zhan.
Over-and underapproximating reach sets for perturbed delay differential equations. IEEE Transactions on Automatic
Control (IEEE TAC), vol.66: 283--290,2020.'</strong></a>

The set-boundary based method can be used to perform reachability analysis for systems modelled by
<ol> 
        <li>ordinary differential equations (subject to Lispchitz continuous perturbations)</li> 
        <li>delay differential equations (subject to Lispchitz continuous perturbations)</li> 
        <li>neural ordinary differential equations</li> 
    </ol> 

# Installation

No need to do installation, just copy the source code into appropriate directory, like using some
Matlab third-party library. We highly recommend you to use `Pycharm` as an IDE for development and
testing.

## Virtual Environment

We recommend that users use miniconda to initialise a virtual environment for the subsequent
installation and running of third party libraries. The steps are as follows.

In the user's current working directory, the user can initialise a virtual test environment called
`pybdr_lab` using the following command.

```shell
conda create -n pybdr_lab
```

After the virtual environment has been initialized, the user activates the virtual test environment
named pybdr_lab that has just been initialized with the following command.

```shell
conda activate pybdr_lab
```

## Dependencies

Now, you can install the necessary third party libraries in this virtual environment using the
following series of commands.

```shell
conda install matplotlib
conda install -c conda-forge numpy
conda install -c conda-forge cvxpy
conda install scipy
conda install -c mosek mosek
pip install pypoman
conda install -c anaconda sympy
pip install open3d
```

For the reason we may use Mosek as a solver for optimisation, we highly recommend you to apply for
an official personal licence, the steps for which can be found
at [this link](https://docs.mosek.com/10.0/licensing/index.html).

# How to use

## Computing Reachable Sets based on Boundary

Example files are provided to show how to use the tool to calculate reachable sets. Users can refer
to the example files provided and modify the dynamics and parameters required for the calculation to
see the effect of using different settings for calculating system reachable sets.

For example, consider following dynamic system:

$$
\begin{align*}
\dot{x} &= y \\
\dot{y} &= (1-x^2)y-x
\end{align*}
$$

```python
import numpy as np

from pyrat.algorithm import ASB2008CDC
from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope, Interval, Geometry
from pyrat.model import *
from pyrat.util.visualization import plot

# init dynamic system
system = NonLinSys(Model(vanderpol, [2, 1]))

# settings for the computation
options = ASB2008CDC.Options()
options.t_end = 6.74
options.step = 0.005
options.tensor_order = 3
options.taylor_terms = 4
options.r0 = [Zonotope([1.4, 2.4], np.diag([0.17, 0.06]))]
options.u = Zonotope.zero(1, 1)
options.u_trans = np.zeros(1)

# settings for the using geometry
Zonotope.REDUCE_METHOD = Zonotope.REDUCE_METHOD.GIRARD
Zonotope.ORDER = 50

# reachable sets computation
ti, tp, _, _ = ASB2008CDC.reach(system, options)

# visualize the results
plot(tp, [0, 1])
```

![](doc/imgs/vanderpol_example.png)

## Frequently Asked Questions and Troubleshooting

### the computation is slow

- The tool supports two modes of computation for reachable sets, one is to compute the reachable set
  of evolved states
  based on the entire initial set in a set propagation manner, and the other is to compute the
  reachable set of
  evolved states based on the boundary of the initial state set.

  There are several reasons for slow computations: large computational time intervals,
  small step,
  high Taylor expansion orders, and a large number of state variables.

  In order to speed up the computations, experiments can be carried out using a smaller computational time
  horizon, a smaller order of
  expansion such as 2, and a larger time step, and then gradually increase the computational time
  horizon and the order
  of expansion according to the results of this setting, in order to obtain the desired set of
  reachable states at a
  suitable time consumption.

### controlling the wrapping effect

<!--  Due to the wrapping effect of set propagation based algorithms, 
it is inevitable that the range of
  the computed
  reachable state sets is too conservative under inappropriate settings, making it impossible to
  obtain a reachable
  state set that meets the requirements.  If these problems occur, -->

- One can further improve the accuracy of the reachable set calculation based on splitting boundaries of initial sets,
  or by reducing the step and
  increasing the
  order of the Taylor expansion.

  <!--then the set explosion problem can be avoided in the computation
  possibly.-->

> Feel free to contact [dingjianqiang0x@gmail.com](mailto:dingjianqiang0x@gmail.com) if you find any
> issues or bugs in this code, or you struggle to run it in any way.

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE.md) file for
details.

## Acknowledgement

This tool has been developed with reference to models in tools such as Flow*, CORA and other
reachable set calculation tools.

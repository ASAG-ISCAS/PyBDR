import numpy as np

"""
Simulator class for simulating trajectories of specified dynamic system
"""


class Simulator:
    def __init__(self):
        # TODO do nothing
        pass

    @classmethod
    def _evaluate_nonlinear(cls, sys, x, u):
        return sys.evaluate((x, u), 'numpy', 0, 0)

    @classmethod
    def _evaluate_linear(cls, sys, x, u):
        return sys.evaluate(x, u)

    @classmethod
    def simulate_one_step(cls, sys, step, x, u):
        # assert x.ndim == 2
        if sys.type == "nonlinear":
            return x + cls._evaluate_nonlinear(sys, x, u) * step
        elif sys.type == "linear_simple":
            return x + sys.evaluate(x, u) * step
        else:
            raise NotImplementedError

    @classmethod
    def _preprocess(cls, sys, x, u):
        if sys.type == "nonlinear":
            x = x if isinstance(x, np.ndarray) else np.asarray(x).reshape(-1)  # must 1d at present
            u = u if isinstance(u, np.ndarray) else np.asarray(u).reshape(-1)  # must 1d at present
            return x, u
        elif sys.type == "linear_simple":
            x = x if isinstance(x, np.ndarray) else np.atleast_2d(x)
            u = u if isinstance(u, np.ndarray) else np.atleast_2d(u)
            assert x.ndim <= 2  # single init point or multiple points
            return x, u
        else:
            raise NotImplementedError

    @classmethod
    def simulate(cls, sys, t_end: float, step: float, x, u):
        x, u = cls._preprocess(sys, x, u)
        steps_num = round(t_end / step)
        assert steps_num >= 1

        result_trajs = [x]
        x_next = x

        for i in range(steps_num):
            x_next = cls.simulate_one_step(sys, step, x_next, u)
            result_trajs.append(x_next)

        if sys.type == "nonlinear":  # uggly code
            result_trajs = [traj.reshape((1, -1)) for traj in result_trajs]

        return result_trajs

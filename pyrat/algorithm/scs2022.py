"""
REF:

Safety Controller Synthesis based on MPC
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sympy import *

from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.misc import Set
from pyrat.model import Model
from .cdc2008 import CDC2008


class SCS2022:
    @dataclass
    class Options:
        x0: np.ndarray = None  # initial state
        vx: Callable[[np.ndarray], float] = None  # reach avoid verification
        target: Callable[[np.ndarray], float] = None  # target region
        step: float = 0  # step size
        max_steps: int = 1e3  # max steps for the computation
        invalid_p: bool = False  # if hard to get some valid p by randomly sampling
        # for potential dynamic model
        dim: int = 0  # dimension of the model

        # for p sampling
        p: Callable[[np.ndarray, np.ndarray], float] = None
        n: int = 5
        max_attempts: int = 30
        low: float = -10
        up: float = 10
        # for running time
        cur_step: int = 0
        cur_x: np.ndarray = 0

        def _validate_misc(self, dim: int):
            assert self.x0.ndim == 1 and self.x0.shape[0] == dim
            return True

        def validation(self):
            assert self.dim > 0
            assert self._validate_misc(self.dim)
            assert self.step > 0
            assert self.vx is not None
            assert self.target is not None
            assert self.up > self.low
            self.cur_x = self.x0
            return True

    @classmethod
    def simulation(cls, f, x, px, opt: Options):
        def func(_x):
            return np.array(f(_x, px))

        next_x = func(x) * opt.step + x
        return next_x

    @classmethod
    def distance(cls, p: np.ndarray, target):
        # about our case, target region is bounded by a circle
        d = p - np.array([0, 0.5])
        # return 0
        return np.linalg.norm(d, ord=2) + np.sqrt(0.1)

    @classmethod
    def get_valid_p(cls, f, opt: Options):
        # sampling potential valid ps
        params = np.random.rand(opt.max_attempts, opt.n)
        params = params * (opt.up - opt.low) + opt.low
        next_xs = [
            cls.simulation(f, opt.cur_x, lambda x: opt.p(x, param), opt)
            for param in params
        ]
        dists = [cls.distance(next_x, opt.target) for next_x in next_xs]
        # get the index of potential p according to potential distance to the target
        indices = np.argsort(dists)
        for idx in indices:  # check every potential p
            is_safe, next_r = cls.safe_control(f, lambda x: opt.p(x, params[idx]), opt)
            if is_safe:
                return params[idx], next_r  # control and next reachable set

        # if none of these potential p is valid, return None
        return None, None

    @classmethod
    def init_model(cls, func, px, opt: Options):
        def _f(x, u):
            return Matrix(func(x, px))

        @dataclass
        class InnerModel(Model):
            vars = symbols(("x:" + str(opt.dim), "u:1"))
            f = _f(*vars)
            name = "inner model"
            dim = f.rows

        return InnerModel()

    @classmethod
    def safe_control(cls, f, px, opt: Options):
        r0 = [Set(Zonotope(opt.cur_x, np.eye(opt.cur_x.shape[0]) * 0.05))]

        # init system according to this given px
        system = NonLinSys.Entity(cls.init_model(f, px, opt))

        # init options for verification
        verif_opt = CDC2008.Options()
        verif_opt.u = Zonotope.zero(1)
        verif_opt.u_trans = np.zeros(1)
        verif_opt.step = opt.step
        verif_opt.t_end = opt.step
        assert verif_opt.validation(opt.dim)
        r_ti, r_tp = CDC2008.reach_one_step(system, r0, verif_opt)
        # check if r_ti inside the RA set according to the value function
        for vertex in r_ti[0].vertices:
            if opt.vx(vertex) > 0:
                return False, None
        return True, r_tp[0].geometry

    @classmethod
    def terminate(cls, opt: Options):
        if opt.cur_step >= opt.max_steps:
            print("MAX STEPS STOP")
            return True
        # check if current state hit the target already
        if opt.target(opt.cur_x) <= 0:
            print(opt.cur_x)
            print("HIT TARGET STOP")
            return True
        # check if hard to get some valid p as controller
        if opt.invalid_p:
            print("INVALID P STOP")
            return True
        # else continue
        return False

    @classmethod
    def one_step_forward(cls, f, opt: Options):
        p, next_r = cls.get_valid_p(f, opt)
        if p is None:
            opt.invalid_p = True
            return None
        # else valid p, update current state with center of the reachable set
        opt.cur_x = next_r.c
        opt.cur_step += 1
        return next_r

    @classmethod
    def synthesis(cls, f, opt: Options):
        assert opt.validation()

        x = [opt.x0]
        rs = []
        while not cls.terminate(opt):
            r = cls.one_step_forward(f, opt)
            if r is not None:
                rs.append(r)
                x.append(opt.cur_x)
        # return the result
        return x, rs

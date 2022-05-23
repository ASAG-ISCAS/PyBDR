"""
REF:

Safety Controller Synthesis based on MPC
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial

from pyrat.dynamic_system import NonLinSys
from pyrat.geometry import Zonotope
from pyrat.misc import Set
from .cdc2008 import CDC2008


class SCS2022:
    @dataclass
    class Options:
        x0: np.ndarray = None
        v: Polynomial = None
        target: callable = None
        px: callable = None  # controller function
        step: float = 0
        max_steps: int = 100
        invalid_p: bool = False
        # for verification
        verif_opt: CDC2008.Options = None
        # for p sampling
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

        def _init_verification_options(self, dim: int):
            self.verif_opt = CDC2008.Options()
            self.verif_opt.t_end = self.step
            self.verif_opt.step = self.step
            return True

        def validation(self, dim: int):
            assert self._validate_misc(dim)
            assert self._init_verification_options(dim)
            assert self.step > 0
            self.cur_x = self.x0
            return True

    @classmethod
    def sos_v(cls, g, h):
        # TODO
        raise NotImplementedError

    @classmethod
    def simulation(cls, sys: NonLinSys.Entity, x, p, step: float):
        next_x = sys.evaluate((x, p)) * step + x
        return next_x

    @classmethod
    def distance(cls, p: np.ndarray, target: Polynomial):
        # TODO
        return 0

    @classmethod
    def get_valid_p(cls, sys: NonLinSys.Entity, opt: Options):
        # sampling potential valid ps
        params = np.random.rand(opt.max_attempts, opt.n)
        next_xs = [
            cls.simulation(sys, opt.cur_x, opt.px(opt.cur_x, param), opt.step)
            for param in params
        ]
        dists = [cls.distance(next_x, opt.target) for next_x in next_xs]
        # get the index of potential p according to potential distance to the target
        indices = np.argsort(dists)
        for idx in indices:  # check every potential p
            p = opt.px(opt.cur_x, params[idx])
            is_safe, next_r = cls.safe_control(sys, p, opt)
            if is_safe:
                return p, next_r  # control and next reachable set

        # if none of these potential p is valid, return None
        return None, None

    @classmethod
    def safe_control(cls, sys: NonLinSys.Entity, p, opt: Options):
        r0 = [Set(Zonotope(opt.cur_x, np.eye(sys.dim) * 0))]
        opt.verif_opt.u = Zonotope(p, np.eye(p.shape[0]) * 0)
        opt.verif_opt.u_trans = p
        assert opt.verif_opt.validation(sys.dim)
        r_ti, r_tp = CDC2008.reach_one_step(sys, r0, opt.verif_opt)
        # check if r_ti inside the RA sets
        for vertex in r_ti[0].vertices:
            if opt.target(vertex) > 0:
                return False, None
        return True, r_tp[0].geometry

    @classmethod
    def terminate(cls, opt: Options):
        if opt.cur_step >= opt.max_steps:
            return True
        # check if current state hit the target already
        if opt.target(opt.cur_x) <= 0:
            return True
        # check if hard to get some valid p as controllerS
        if opt.invalid_p:
            return True
        # else continue
        return False

    @classmethod
    def one_step_forward(cls, sys: NonLinSys.Entity, opt: Options):
        p, next_r = cls.get_valid_p(sys, opt)
        if p is None:
            opt.invalid_p = True
            return
        # else valid p, update current state with center of the reachable set
        opt.cur_x = next_r.c

    @classmethod
    def synthesis(cls, sys: NonLinSys.Entity, opt: Options):
        assert opt.validation(sys.dim)

        x = []
        while not cls.terminate(opt):
            x.append(opt.cur_x)
            cls.one_step_forward(sys, opt)

        # return the result
        return x

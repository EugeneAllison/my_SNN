#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["ode_solver", "ODESystem"]

import os
import sys
import math
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from pyutils.tqdm import tqdm, trange


def ode_solver(
        func, x0, t0, t1,
        n_steps=1, solver="euler", post_process=None,
        func_args=[], func_kwargs={}, return_error=False):
    '''
    versatile ode solver

    Args:
        func: dx/dt = func(x, t, *func_args, **func_kwargs)
        x0 (np.ndarray): initial state
        t0 (scalar): initial time
        t1 (scalar): terminal time
    Returns:
        np.ndarray: list of time
        np.ndarray: list of the dynamics
    '''
    h = (t1 - t0) / n_steps
    a_coef, b_coef, c_coef, extended = coef_dict[solver]
    dim = a_coef.shape[0]
    if extended and return_error:
        d_coef = b_coef[1] - b_coef[0]

    error_list = []
    t_list, x_list = np.zeros(n_steps + 1), np.zeros((n_steps + 1, *x0.shape))

    t_now = t0
    x_now = x0
    t_list[0] = t_now
    x_list[0] = x_now
    for _ in range(1, n_steps + 1):
        t_pre = t_now
        k_list = [None] * dim
        for _i in range(dim):
            x_calc = x_now
            for _j in range(_i):
                if abs(a_coef[_i, _j]) > 0:
                    x_calc = x_calc + (h * a_coef[_i, _j]) * k_list[_j]
            if (post_process is not None) and (_i > 0):
                x_calc = post_process(x_calc)
            t_now = t_pre + h * c_coef[_i]
            if (_i == dim - 1) and extended and (not return_error):
                break
            k_list[_i] = func(x_calc, t_now, *func_args, **func_kwargs)
        if extended:
            x_now = x_calc
            if return_error:
                x_err = sum((h * d_coef[_i]) * k_list[_i]
                            for _i in range(dim))
                error_list.append(x_err)
        else:
            for _i in range(dim):
                x_now = x_now + (h * b_coef[0, _i]) * k_list[_i]
            t_now = t_pre + h
        t_list[_] = t_now
        x_list[_] = x_now
    if return_error and extended:
        return (t_list, x_list), error_list
    else:
        return (t_list, x_list)


coef_dict = {
    "euler": [
        np.array([[0]]),
        np.array([[1]]),
        np.array([1]),
        False
    ],
    "heun": [
        np.array([[0, 0],
                  [1, 0]]),
        np.array([[1 / 2, 1 / 2]]),
        np.array([0, 1]),
        False
    ],
    "rk2": [
        np.array([[0, 0],
                  [1 / 2, 0]]),
        np.array([[0, 1]]),
        np.array([0, 1 / 2]),
        False
    ],
    "rk23": [
        np.array([
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 3 / 4, 0, 0],
            [2 / 9, 1 / 3, 4 / 9, 0]]),
        np.array([
            [2 / 9, 1 / 3, 4 / 9, 0],
            [7 / 24, 1 / 4, 1 / 3, 1 / 8]]),
        np.array([0, 1 / 2, 3 / 4, 1]),
        True
    ],
    "rk4": [
        np.array([
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [0, 0, 1, 0]]),
        np.array([
            [1 / 6, 1 / 3, 1 / 3, 1 / 6]]),
        np.array([0, 1 / 2, 1 / 2, 1]),
        False
    ],
    "rk45dp": [
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561,
                -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247,
                49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192,
                -2187 / 6784, 11 / 84, 0]]),
        np.array([
            [35 / 384, 0, 500 / 1113, 125 / 192,
                -2187 / 6784, 11 / 84, 0],
            [5179 / 57600, 0, 7571 / 16695, 393 / 640,
                -92097 / 339200, 187 / 2100, 1 / 40]]),
        np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]),
        True
    ]
}


class ODESystem(object):
    def __init__(self, func, x0, t0, solver="rk45dp"):
        self.func = func
        self.state = np.array(x0)
        self.time = t0
        self.solver = solver

    def step(self, dt, T=None, solver=None, **kwargs):
        if T is None:
            T = dt
        if solver is None:
            solver = self.solver
        t_list, x_list = ode_solver(
            self.func, self.state,
            self.time, self.time + T, n_steps=int(T / dt),
            solver=solver, func_kwargs=kwargs)
        self.state[:] = x_list[-1]
        self.time = t_list[-1]
        return t_list, x_list

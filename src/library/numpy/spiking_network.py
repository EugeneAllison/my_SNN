#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import copy
import joblib
import operator
import argparse
import itertools
import functools
import scipy.stats
import numpy as np
import pandas as pd

from functools import reduce
from collections import defaultdict


# 全局参数
SQUASH = [0.9595, 0.0661]
ALPHA = SQUASH[0] * SQUASH[1]
V_AVG = 8
V_SD = 10
A_AVG = ALPHA * V_AVG
A_SD = ALPHA * V_SD


"""
阈值电位（h_th）：超过该值时，神经元触发尖峰。
窗口期（refs）：在尖峰触发后，神经元需要一个恢复期，期间不会再次触发尖峰。

v_sd (Standard Deviation of Voltage) 神经元膜电位标准差
iota (Adjustment for Weight Update) 调整因子，用于权重更新
v_sm (smoothed???)

"""


class SpikingLayer(object):
    def __init__(self, dim_in, dim_out, tau=20, h_th=0.4):
        self.dim_in, self.dim_out = dim_in, dim_out
        self.tau, self.h_th = 20, 0.4

    @staticmethod
    def __random_weight(N, v_avg=V_AVG, v_sd=V_SD, b_avg=0.8, alpha=ALPHA):
        v_sm = v_sd * v_sd + v_avg * v_avg
        # v2 = v_sigma + v1**2
        W_avg = (v_avg - b_avg) / (alpha * N * v_avg)
        # W1 = (v1 - b1) / (alpha * N * v1)
        W_sm = (
            v_sm
            + alpha**2 * (N - N**2) * W_avg**2 * v_avg**2
            - 2 * alpha * N * b_avg * v_avg * W_avg
            - b_avg**2
        ) / (alpha**2 * N * v_sm)
        # W2 = (v2 + alpha**2 * (N - N**2) * W1**2 * v1**2 - 2 * alpha * N * b1 * v1 * W1 - b1**2) / (alpha**2 * N * v2)
        W_sd = math.sqrt(W_sm - W_avg**2)
        # W_sigma = W2 - W1 ** 2
        return b_avg, W_avg, W_sd
    # 根据输入的统计特性计算权重的均值（W_avg）和标准差（W_sd）。

    def init_weight(self, **kwargs):
        b_avg, W_avg, W_sd = self.__random_weight(self.dim_in)
        self.W = (
            np.random.uniform(
                low=-math.sqrt(3), high=math.sqrt(3), size=(self.dim_in, self.dim_out)
            )
            * W_sd
            + W_avg
        )
        self.b = np.ones(self.dim_out) * b_avg

        self._B = (
            np.random.uniform(
                low=-math.sqrt(3), high=math.sqrt(3), size=(self.dim_out, self.dim_in)
            )
            * W_sd
            + W_avg
        )
        self.eta = 2 / self.dim_in

    def init_state(self, inputs, keep_state=True):
        shape = (*inputs.shape[:-1], self.dim_out)
        if hasattr(self, "v") and shape == self.v.shape and keep_state:
            return
        self.v = np.zeros(shape)
        self.h = np.zeros(shape)
        self.refs = np.zeros(shape)

    def forward(self, dt, inputs):
        self.a = inputs
        self.v = np.dot(self.a, self.W) + self.b
        self.refs[self.refs > 0] += dt
        self.refs[self.refs >= 1] = 0
        is_ref = self.refs > 0
        cv = dt / self.tau
        ch = 1 - cv
        self.h = (~is_ref) * (ch * self.h + cv * self.v)
        self.refs[np.logical_and(self.h > self.h_th, ~is_ref)] += 1e-8
        return np.logical_and(self.refs > 0, self.refs < 1.0)

    def backward(self, dt, e, bfunc, lr=1.0):
        if hasattr(self, "B"):
            delta = np.dot(e, self.B)
        else:
            delta = e
        g = bfunc(self.v)
        iota = lr * delta * g
        self.W = self.W - np.dot(self.a.T, iota)
        self.b = self.b - iota

    def backprop(self, dt, delta, bfunc, lr=1.0):
        g = bfunc(self.v)
        iota = delta * g
        bp = iota.dot(self.W.T)
        self.W = self.W - lr * np.dot(self.a.T, iota)
        self.b = self.b - lr * iota
        return bp

    def record(self, bmin=-5, bmax=5, bcount=201, size=None):
        if not hasattr(self, "hist"):
            self.bins = np.linspace(bmin, bmax, bcount)
            self.center = self.bins[1:] - (self.bins[1] - self.bins[0])
            if size is None:
                self.hist = []
            else:
                self.hist = np.zeros((size, *self.center.shape))
            self.update_cnt = 0
        lw = np.log10(np.abs(self.W))
        hist, _ = np.histogram(lw, self.bins)
        hist = hist.astype(float)
        hist[hist == 0] = np.nan
        if type(self.hist) is list:
            self.hist.append(hist)
        else:
            self.hist[self.update_cnt % self.hist.shape[0]] = hist
        self.update_cnt += 1

    @property
    def histogram(self):
        return np.asarray(self.hist[: self.update_cnt])


class SpikingNetwork(object):
    def __init__(self, dims, bf=0.0338):
        self.layers = []
        class_num = dims[-1]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            l = SpikingLayer(dim_in, dim_out)
            l.init_weight()
            self.layers.append(l)

        B = np.eye(class_num)
        self.layers[-1].B = np.array(B)
        for l_pre, l_post in zip(reversed(self.layers[:-1]), reversed(self.layers[1:])):
            print(B.shape, l_post._B.shape)
            B = np.dot(B, l_post._B) * bf
            l_pre.B = B
        for idx in range(0, len(self.layers)):
            self.layers[idx].B *= self.layers[idx].eta

    def reset(self, inputs, **kwargs):
        for l in self.layers:
            l.init_state(inputs, **kwargs)

    def step(
        self, dt, inputs, outputs, bfunc=None, lr=1.0, final_only=False, use_bp=False
    ):
        a = inputs
        for l in self.layers:
            a = l.forward(dt, a)

        e = a - outputs
        if bfunc is not None:
            if final_only:
                self.layers[-1].backward(dt, e, bfunc, lr)
            elif use_bp:
                delta = e
                for l in self.layers[::-1]:
                    delta = l.backprop(dt, delta * 0.0338, bfunc, lr)
            else:
                for l in self.layers[::-1]:
                    l.backward(dt, e, bfunc, lr)
        return a

    def record(self, **kwargs):
        for l in self.layers:
            l.record(**kwargs)

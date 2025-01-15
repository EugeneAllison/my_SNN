#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局参数
SQUASH = [0.9595, 0.0661]
ALPHA = SQUASH[0] * SQUASH[1]
V_AVG = 8
V_SD = 10
A_AVG = ALPHA * V_AVG
A_SD = ALPHA * V_SD


class SpikingLayer(torch.nn.Module):

    def __init__(self, dim_in, dim_out, tau=20, h_th=0.4, device=device):
        # super(SpikingLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.tau = tau
        self.h_th = h_th
        self.device = device

        # self.W = None
        # self.b = None
        # self.B = None
        # self.v = None
        # self.h = None
        # self.refs = None
        # self.eta = None

    def init_weight(self, **kwargs):
        b_avg, W_avg, W_sd = self.__random_weight(self.dim_in)
        self.W = (
            torch.rand(self.dim_in, self.dim_out, device=self.device) * 2 - 1
        ) * math.sqrt(3) * W_sd + W_avg
        self.b = torch.ones(self.dim_out, device=self.device) * b_avg

        self._B = (
            torch.rand(self.dim_out, self.dim_in, device=self.device) * 2 - 1
        ) * math.sqrt(3) * W_sd + W_avg
        self.eta = 2 / self.dim_in

    # def init_state(self, inputs, keep_state=True):
    #     shape = (*inputs.shape[:-1], self.dim_out)
    #     if hasattr(self, "v") and shape == self.v.shape and keep_state:
    #         return
    #     self.v = torch.zeros(shape, device=self.device)
    #     self.h = torch.zeros(shape, device=self.device)
    #     self.refs = torch.zeros(shape, device=self.device)

    def init_state(self, inputs, keep_state=True):
        shape = (*inputs.shape[:-1], self.dim_out)
        if getattr(self, "v", None) is not None and shape == self.v.shape and keep_state:
            return
        self.v = torch.zeros(shape, device=self.device)
        self.h = torch.zeros(shape, device=self.device)
        self.refs = torch.zeros(shape, device=self.device)

    def forward(self, dt, inputs):
        # self.a = inputs
        self.a = inputs.to(
            dtype=self.W.dtype
        )  # 将 inputs 的数据类型转换为与 self.W 一致
        self.v = torch.matmul(self.a, self.W) + self.b
        self.refs[self.refs > 0] += dt
        self.refs[self.refs >= 1] = 0
        is_ref = self.refs > 0
        cv = dt / self.tau
        ch = 1 - cv
        self.h = (~is_ref) * (ch * self.h + cv * self.v)
        self.refs[(self.h > self.h_th) & (~is_ref)] += 1e-8
        return (self.refs > 0) & (self.refs < 1.0)

    def backward(self, dt, e, bfunc, lr=1.0):
        if hasattr(self, "B"):
            delta = torch.matmul(e, self.B)
        else:
            delta = e
        g = bfunc(self.v)
        iota = lr * delta * g
        self.W = self.W - torch.matmul(self.a.T, iota)
        self.b = self.b - iota

    def backprop(self, dt, delta, bfunc, lr=1.0):
        g = bfunc(self.v)
        iota = delta * g
        bp = torch.matmul(iota, self.W.T)
        self.W = self.W - lr * torch.matmul(self.a.T, iota)
        self.b = self.b - lr * iota
        return bp

    def record(self, bmin=-5, bmax=5, bcount=201, size=None):
        if not hasattr(self, "hist"):
            self.bins = torch.linspace(bmin, bmax, bcount, device=self.device)
            self.center = self.bins[1:] - (self.bins[1] - self.bins[0])
            if size is None:
                self.hist = []
            else:
                # self.hist = torch.zeros((size, *self.center.shape), device=self.device)
                self.hist = torch.zeros(
                    (size, bcount - 1), device=self.device
                )  # 保持与 hist 一致
            self.update_cnt = 0
        lw = torch.log10(torch.abs(self.W))
        hist = torch.histc(lw, bins=bcount, min=bmin, max=bmax)
        hist = hist[:-1]  # 确保与 self.hist 的形状一致, new line, 之前没有这行
        hist[hist == 0] = float("nan")
        if isinstance(self.hist, list):
            self.hist.append(hist.cpu().numpy())
        else:
            self.hist[self.update_cnt % self.hist.shape[0]] = hist
        self.update_cnt += 1

    @property
    def histogram(self):
        if isinstance(self.hist, list):
            return torch.tensor(self.hist[: self.update_cnt], device=self.device)
        return self.hist[: self.update_cnt]

    @staticmethod
    def __random_weight(N, v_avg=V_AVG, v_sd=V_SD, b_avg=0.8, alpha=ALPHA):
        v_sm = v_sd**2 + v_avg**2
        W_avg = (v_avg - b_avg) / (alpha * N * v_avg)
        W_sm = (
            v_sm
            + alpha**2 * (N - N**2) * W_avg**2 * v_avg**2
            - 2 * alpha * N * b_avg * v_avg * W_avg
            - b_avg**2
        ) / (alpha**2 * N * v_sm)
        W_sd = math.sqrt(W_sm - W_avg**2)
        return b_avg, W_avg, W_sd


class SpikingNetwork(object):

    def __init__(self, dims, bf=0.0338, device=device):
        self.layers = []
        class_num = dims[-1]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layer = SpikingLayer(dim_in, dim_out, device=device)
            layer.init_weight()
            self.layers.append(layer)

        B = torch.eye(class_num, device=device)
        self.layers[-1].B = B.clone()
        for l_pre, l_post in zip(reversed(self.layers[:-1]), reversed(self.layers[1:])):
            print(B.shape, l_post._B.shape)
            # B = np.dot(B, l_post._B) * bf
            # B = torch.matmul(B, l_post.B.T) * bf
            B = torch.matmul(B, l_post._B) * bf
            l_pre.B = B.clone()
        for layer in self.layers:
            layer.B *= layer.eta

    def reset(self, inputs, **kwargs):
        for layer in self.layers:
            layer.init_state(inputs, **kwargs)

    def step(
        self, dt, inputs, outputs, bfunc=None, lr=1.0, final_only=False, use_bp=False
    ):
        a = inputs
        for layer in self.layers:
            a = layer.forward(dt, a)

        # 确保 a 和 outputs 是数值类型
        a = a.to(dtype=torch.float32)  # 将 a 转换为浮点类型
        outputs = outputs.to(dtype=torch.float32)  # 将 outputs 转换为浮点类型
        e = a - outputs
        if bfunc is not None:
            if final_only:
                self.layers[-1].backward(dt, e, bfunc, lr)
            elif use_bp:
                delta = e
                for layer in reversed(self.layers):
                    delta = layer.backprop(dt, delta * 0.0338, bfunc, lr)
            else:
                for layer in reversed(self.layers):
                    layer.backward(dt, e, bfunc, lr)
        return a

    def record(self, **kwargs):
        for layer in self.layers:
            layer.record(**kwargs)

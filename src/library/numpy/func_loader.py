#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "tanh",
    "dtanh",
    "opto",
    "dopto",
    "gaussian",
    "accurate",
    "acc_clip",
    "approx",
    "sine",
    "cosine",
    "affine",
    "heaviside",
    "triangle",
    "square",
    "sign",
    "linear",
    "constant",
    "NoiseFunc",
]

"""
__all__ 是一个特殊的 Python 列表，定义了模块的公共接口，即在使用 from 模块 import * 时，允许导入的名称。
如果没有定义 __all__，默认会导入所有非下划线开头的全局变量和函数。

示例：
定义了 __all__ 后，from 模块 import * 仅会导入 __all__ 中列出的内容。
在这个例子中，__all__ 定义了 18 个函数和 1 个类 NoiseFunc。

sys：用于与 Python 解释器交互，例如路径管理。
"""

import sys
import math

import numpy as np

sys.path.append(".")

from src.library.numpy.spiking_network import SQUASH


def tanh(t):
    return np.tanh(t)


def dtanh(t):
    return 1 - np.tanh(t) ** 2


def opto(t):
    return np.cos(t) ** 2


def dopto(t):
    return -np.sin(2 * t)


def gaussian(t):
    return np.exp(-(t**2))


def accurate(v, t_ref=1, tau=20, h_th=0.4):
    with np.errstate(divide="ignore", invalid="ignore"):
        ans = np.array(v)
        ans = (
            h_th
            * t_ref
            * tau
            / (v * (v - h_th) * (t_ref + tau * np.log(v / (v - h_th))) ** 2)
        )
        ans[v <= h_th] = 0
    return ans


"""
根据公式计算准确度函数，适用于神经网络的模拟。
使用了 NumPy 的错误状态控制（如零除处理）。

"""


def acc_clip(v, clip_max=1.0, **kwargs):
    return np.clip(0, clip_max, accurate(v, **kwargs))


def approx(t):
    return (t > 0) * (1 / np.cosh(t)) ** 2


def sine(t):
    return np.sin(t)


def cosine(t):
    return np.sin(t)


def affine(t, phase=0.0, amp=1.0):
    return amp * t + phase


def heaviside(t):
    return np.heaviside(t, 0)


"""
单位阶跃函数，又称赫维赛德阶跃函数
参数为负时值为0，参数为正时值为1。
"""


def constant(t):
    return np.ones_like(t)


"""
constant(t)：返回与 t 形状一致的全 1 数组。

"""


def linear(t):
    return t


def triangle(t):
    # 返回三角波函数，形状为三角形的周期函数。
    return 1 - np.abs(np.clip(t, -1, 1))


def square(t):
    return (np.abs(t) < 1).astype(float)


def sign(t):
    return np.clip(t, -1, 1)


class NoiseFunc(object):
    def __init__(self, func, noise_amp):
        self.func = func
        self.noise_amp = noise_amp
        # noise_amp：噪声的振幅。amplitude

    def __call__(self, t):
        noise = np.random.randn(t.size())
        noise = self.noise_amp * noise.to(t)
        return self.func(t + noise)

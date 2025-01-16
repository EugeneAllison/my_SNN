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
    "generate_random_ga_01",
]

import sys
import math
import torch

sys.path.append(".")

from src.library.numpy.spiking_network import SQUASH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tanh(t):
    return torch.tanh(t)


def dtanh(t):
    return 1 - torch.tanh(t) ** 2


def opto(t):
    return torch.cos(t) ** 2


def dopto(t):
    return -torch.sin(2 * t)


def gaussian(t):
    return torch.exp(-(t**2))


def accurate(v, t_ref=1, tau=20, h_th=0.4):
    ans = torch.clone(v)
    valid_mask = v > h_th
    log_term = torch.log(v / (v - h_th))
    denominator = v * (v - h_th) * (t_ref + tau * log_term) ** 2
    ans[valid_mask] = h_th * t_ref * tau / denominator[valid_mask]
    ans[~valid_mask] = 0
    return ans


def acc_clip(v, clip_max=1.0, **kwargs):
    return torch.clamp(accurate(v, **kwargs), min=0, max=clip_max)


def approx(t):
    return (t > 0).float() * (1 / torch.cosh(t)) ** 2


def sine(t):
    return torch.sin(t)


def cosine(t):
    return torch.cos(t)


def affine(t, phase=0.0, amp=1.0):
    return amp * t + phase


def heaviside(t):
    return torch.heaviside(t, values=torch.tensor(0.0))


def constant(t):
    return torch.ones_like(t)


def linear(t):
    return t


def triangle(t):
    return 1 - torch.abs(torch.clamp(t, -1, 1))


def square(t):
    return (torch.abs(t) < 1).float()


def sign(t):
    return torch.clamp(t, -1, 1)


class NoiseFunc(object):
    def __init__(self, func, noise_amp):
        self.func = func
        self.noise_amp = noise_amp

    def __call__(self, t):
        noise = self.noise_amp * torch.randn_like(t)
        return self.func(t + noise)

def generate_random_ga_01(shape, ratio, seed):
    """生成指定比例为1的随机向量"""
    torch.manual_seed(seed)
    ga = torch.zeros(shape, device=device)
    num_ones = int(ratio * shape[-1])
    indices = torch.randperm(shape[-1], device=device)[:num_ones]
    ga[:, indices] = 1
    return ga


"""
代码修改说明：

1. 将 `numpy` 替换为 `torch`：
   - NumPy 中的 `np.tanh` 改为 PyTorch 的 `torch.tanh`。
   - NumPy 中的 `np.exp` 改为 PyTorch 的 `torch.exp`。
   - 等效替换所有函数和方法，例如 `np.cosh` 替换为 `torch.cosh`。

2. 张量操作：
   - NumPy 数组用 `torch.Tensor` 替换。
   - 使用 PyTorch 的 `torch.clone` 创建可修改的张量副本。
   - 使用 `torch.clamp` 替换 `np.clip` 实现限制。

3. 特殊函数：
   - `np.heaviside` 替换为 `torch.heaviside`。
   - NumPy 的随机数生成 `np.random.randn` 替换为 `torch.randn_like`。

4. 噪声生成：
   - 在 `NoiseFunc` 中生成与输入张量形状一致的噪声，使用 PyTorch 的 `torch.randn_like`。

这些修改确保代码功能和逻辑不变，同时改用 PyTorch 以支持 GPU 加速。"""

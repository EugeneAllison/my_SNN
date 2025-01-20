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
    "generate_random_ga",
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


def generate_random_ga(shape, n, seed):
    """Generate a random vector with values in the range [-n, n]."""
    torch.manual_seed(seed)
    return (torch.rand(shape) * 2 * n) - n

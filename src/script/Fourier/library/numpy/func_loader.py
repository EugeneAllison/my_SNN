#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "tanh", "dtanh", "opto", "dopto", "gaussian",
    "accurate", "acc_clip", "approx", "sine", "cosine", "affine",
    "heaviside", "triangle", "square", "sign",
    "linear", "constant", "NoiseFunc"]


import sys
import math


import numpy as np

sys.path.append(".")



def tan(t):
    return np.tan(t)


def tanh(t):
    return np.tanh(t)+1


def dtanh(t):
    return 1 - np.tanh(t) ** 2

def opto_sin(t):
    return np.sin(t) ** 2

def opto_tan(t):
    return np.tan(t) ** 2

def opto_tanh(t):
    return np.tanh(t) ** 2

def opto(t):
    return np.cos(t) ** 2


def dopto(t):
    return -np.sin(2 * t)


def gaussian(t):
    return np.exp(-t ** 2)


def gaussian_normalize(t, u=0, v=1):
    a = 1/(v * np.sqrt(2 * np.pi))
    b = u
    c = v
    down = 2 * (c ** 2)
    up = (t-b) ** 2
    return a * np.exp(- up / down)

def gaussian_optimize(t, a=1, b=0, c=1):
    down = 2 * (c ** 2)
    up = (t-b) ** 2
    return a * np.exp(- up / down)

def gaussian_paper(t, a=1, b=0, c=1):
    up = np.abs(t-b)
    return a * np.exp(- c * up)



def accurate(v, t_ref=1, tau=20, h_th=0.4):
    with np.errstate(divide="ignore", invalid="ignore"):
        ans = np.array(v)
        ans = h_th * t_ref * tau / (v * (v - h_th) * (t_ref + tau * np.log(v / (v - h_th))) ** 2)
        ans[v <= h_th] = 0
    return ans #E(a)的精确的导数


def acc_clip(v, clip_max=1, **kwargs):
    return np.clip(0, clip_max, accurate(v, **kwargs))
    #将E的导数进行二进制的离散化

def approx(t):
    return (t > 0) * (1 / np.cosh(t))**2 # check performance E的近似后的导数


def sine(t):
    return np.sin(t)


def cosine(t):
    return np.cos(t)


def affine(t, phase=0.0, amp=1.0):
    return amp * t + phase


def heaviside(t):
    return np.heaviside(t, 0)


def constant(t):
    return np.ones_like(t)


def linear(t):
    return t


def triangle(t):
    return 1 - np.abs(np.clip(t, -1, 1))


def square(t):
    return (np.abs(t) < 1).astype(float)


def sign(t):
    return np.clip(t, -1, 1)


# def random_Fourier_series(t, K)
class Fourier(object):
    def __init__(self, dim, seed=None, resolution=10001):
        self.dim = dim
        self.rnd = np.random.RandomState(seed)
        self.coefs = self.rnd.uniform(-1, 1, dim * 2)
        self.coefs /= np.abs(self.coefs).sum()
        self.resolution = resolution

    def __call__(self, t):
        return self._func(t) - self.m

    def _func(self, t):
        fs = np.pi * np.arange(1, self.dim + 1)
        ot = np.outer(t, fs)
        st = np.sin(ot).dot(self.coefs[:self.dim])
        ct = np.cos(ot).dot(self.coefs[self.dim:])
        out = st + ct
        return out.reshape(*t.shape)

    @property
    def m(self):
        if hasattr(self, "_m"):
            return self._m
        ts = np.linspace(-1, 1, self.resolution)
        ys = self._func(ts)
        self._m = np.min(ys)
        return self._m

class NoiseFunc(object):
    def __init__(self, func, noise_amp):
        self.func = func
        self.noise_amp = noise_amp

    def __call__(self, t):
        noise = np.random.randn(t.size())
        noise = self.noise_amp * noise.to(t)
        return self.func(t + noise)

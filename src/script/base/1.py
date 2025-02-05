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
import torch
import pandas as pd

from functools import reduce
from collections import defaultdict
from sklearn.datasets import fetch_openml
from torchvision import datasets, transforms

sys.path.append(".")

from pyutils.figure import Figure  # 图片
from pyutils.tqdm import tqdm, trange

import src.library.style
import src.library.numpy.func_loader_torch
from src.library.numpy.spiking_network_torch import SpikingNetwork

# # 定义网络的结构
# dims = [3, 5, 6, 4]

# # 初始化网络
# network = SpikingNetwork(dims=dims)

# # 打印每层之间的随机反馈矩阵 B
# for i, layer in enumerate(network.layers):
#     if hasattr(layer, "B"):
#         print(f"Layer {i} B matrix:")
#         print(layer.B)

"""
# 设置设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 随机种子，保证结果可复现
torch.manual_seed(42)


dims = [3, 5, 6, 4]
target_radius = 0.5  # 目标谱半径
seed = 42  # 随机种子

# 初始化网络
network = SpikingNetwork(
    dims=dims, bf=0.0338, target_radius=target_radius, seed=seed
)

# 打印每层的反馈矩阵 B
for i, layer in enumerate(network.layers):
    if hasattr(layer, "B"):
        print(f"\nLayer {i} Final B matrix:")
        print(layer.B)

"""


import torch

# 定义矩阵
B = torch.tensor([[1.0, 2.0], [3.0, 4.0]])


# 创建类并调用函数
class MatrixController:
    def control_spectral_radius(self, matrix, target_radius):
        """
        控制矩阵的谱半径
        使用 SVD (奇异值分解) 计算谱半径，而不是通过 B^T * B。
        """
        # 使用 SVD 获取奇异值
        u, s, v = torch.linalg.svd(matrix)
        spectral_radius = torch.max(s)  # 最大奇异值即为谱半径
        print(f"Original Spectral Radius: {spectral_radius.item():.4f}")

        # 调整矩阵的谱半径
        adjusted_matrix = (matrix / spectral_radius) * target_radius
        u_adj, s_adj, v_adj = torch.linalg.svd(adjusted_matrix)
        new_spectral_radius = torch.max(s_adj)
        print(f"Adjusted Spectral Radius: {new_spectral_radius.item():.4f}")
        return adjusted_matrix


controller = MatrixController()
adjusted_B = controller.control_spectral_radius(B, 1.5)  # 将谱半径调整为 1.5
print("Adjusted Matrix:\n", adjusted_B)


a = torch.tensor([[0.2745, 0.5489], [0.8234, 1.0979]])
u, s, v = torch.linalg.svd(a)
spectral_radius = torch.max(s)
print(int(spectral_radius))
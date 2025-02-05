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

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, required=True)  # 必须
parser.add_argument("--init_id", type=int, default=0)
parser.add_argument("--trial_num", type=int, default=1)
parser.add_argument("--net_dims", type=int, nargs="+", default=[784, 1000, 1000, 10])
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--bfunc", type=str, required=True)  # 必须
parser.add_argument("--amp", type=float, default=0.15)
parser.add_argument("--phase", type=float, default=155)

parser.add_argument("--radian", action="store_true")
parser.add_argument("--final_only", action="store_true")
parser.add_argument("--use_bp", action="store_true")

parser.add_argument("--T", type=float, default=100)
parser.add_argument("--T_th", type=float, default=20)
parser.add_argument("--dt", type=float, default=0.25)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure data is moved to the same device
data_train = datasets.MNIST("../data/mnist", train=True, download=True)
data_eval = datasets.MNIST("../data/mnist", train=False, download=True)

X_train, Y_train = data_train.data.float().to(device) / 255, data_train.targets.to(
    device
)
X_eval, Y_eval = data_eval.data.float().to(device) / 255, data_eval.targets.to(device)
X_train = X_train.view(X_train.shape[0], -1)
X_eval = X_eval.view(X_eval.shape[0], -1)


def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False


func = getattr(src.library.numpy.func_loader_torch, args.bfunc)
if args.radian:
    phase = args.phase
else:
    phase = args.phase * math.pi / 180

bfunc = lambda v: func(args.amp * v + phase)


def run(model, X, Y, dt, T, T_th=float("inf"), callback=None):
    data_size = X.shape[0]
    ts = torch.arange(0, T, dt, device=device)
    ids = torch.randperm(data_size, device=device)
    border_id = 0 if T_th > T else int(T_th / dt)
    pbar = trange(0, data_size, args.batch_size, leave=False)
    correct_cnt = 0

    for i in pbar:
        x = X[ids[i : i + args.batch_size]]
        d = Y[ids[i : i + args.batch_size]]
        dh = torch.eye(10, device=device)[d]

        model.reset(x, keep_state=True)
        os = torch.zeros((ts.shape[0], *dh.shape), device=device)
        for cnt, t in enumerate(ts):
            if t < T_th:
                out = model.step(dt, x, dh)
            else:
                out = model.step(
                    dt, x, dh, bfunc, args.lr, args.final_only, args.use_bp
                )
            os[cnt] = out
        if callback is not None:
            callback(model)
        o = torch.argmax(os[border_id:].sum(dim=0), dim=-1)
        acc = (o == d).float().mean().item()
        correct_cnt += int((o == d).sum())
        pbar.set_description("acc={:.2f}".format(acc))
    return model, float(correct_cnt / data_size)


if __name__ == "__main__":
    os.makedirs(args.root_dir, exist_ok=True)

    with open(f"{args.root_dir}/args.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    # model = SpikingNetwork(args.net_dims).to(device)
    model = SpikingNetwork(args.net_dims)

    minibatch_cnt = X_train.shape[0] // args.batch_size
    model.record(size=args.n_epochs * minibatch_cnt * 2)

    fig_hist = Figure(figsize=(8 * len(model.layers), 8))  # 图片 137（now）～～147
    fig_hist.create_grid((1, len(model.layers)))

    def display(model):
        model.record()
        for i, l in enumerate(model.layers):
            fig_hist[i].cla()
            fig_hist[i].plot_matrix(
                l.histogram,
                num_label_x=10,
                ticks_fmt="{:.0f}",
                x=l.center,
                colorbar=False,
                aspect="auto",
            )
        fig_hist.savefig(f"{args.root_dir}/historgram.png")

    records = defaultdict(list)
    for epoch in trange(args.n_epochs, leave=True):
        # 图片：以下到fig.close
        model, acc_t = run(
            model, X_train, Y_train, args.dt, args.T, args.T_th, callback=display
        )
        records["acc_t"].append(acc_t)
        model, acc_e = run(model, X_eval, Y_eval, args.dt, args.T)
        records["acc_e"].append(acc_e)
        fig = Figure()
        fig[0].plot(records["acc_t"])
        fig[0].plot(records["acc_e"])
        fig[0].set_title(
            "best: {:.4f}/{:.4f}".format(max(records["acc_t"]), max(records["acc_e"]))
        )
        fig.savefig(f"{args.root_dir}/records.png")
        fig.close()

        histogram = {str(i): l.histogram for i, l in enumerate(model.layers)}
        torch.save(histogram, f"{args.root_dir}/histogram.pt")

    with open(f"{args.root_dir}/records.pkl", mode="wb") as f:
        joblib.dump(records, f)

    with open(f"{args.root_dir}/records.json", mode="w") as f:
        json.dump(records, f, indent=4)

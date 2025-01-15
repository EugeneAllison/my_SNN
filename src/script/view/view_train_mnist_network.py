#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import glob
import parse
import joblib
import argparse
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torchvision import datasets, transforms

sys.path.append(".")

from src.library.numpy.func_loader import *
from pyutils.figure import Figure
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.ticker as mticker


matplotlib.rcParams["pdf.fonttype"] = 42
sns.set(font_scale=1.5, font="Segoe UI")
sns.set_palette("tab10")
sns.set_style("whitegrid", {'grid.linestyle': '--'})

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='../result/train_mnist_network')
parser.add_argument("--load_file", type=str, default="records.pkl")
parser.add_argument("--regex", type=str, default=".*")
parser.add_argument("--figsize", type=float, nargs=2, default=[40 / 3, 7.5])
parser.add_argument("--ci", type=str, default="sd")
parser.add_argument("--architecture", type=str, default="dfa")
parser.add_argument("--amp", type=str, default="*")
parser.add_argument("--phase", type=str, default="*")
parser.add_argument("--hide_legend", action="store_true")
parser.add_argument("--use_cache", action="store_true")
parser.add_argument("--ylim", type=float, nargs="+", default=[None, 100])
parser.add_argument('--save_dir', type=str, default=None)

args = parser.parse_args()


def approx(t):
    return (t > 0) * (1 / np.cosh(t))**2


def opto(t, amp=1.0, phase=0):
    return np.cos(amp * t + np.pi * phase / 180) ** 2


def load_data(_dir):
    *_, label, _ = list(filter(None, _dir.split("/")))
    if not re.match(args.regex, label):
        return None
    file_path = "{}/{}".format(_dir, args.load_file)
    if not os.path.exists(file_path):
        return None
    with open(file_path, mode="rb") as f:
        result = joblib.load(f)
    return label, result


def load_performance(ffunc, architecture="dfa", amp="*", phase="*", lr="*"):
    dir_list = glob.glob(
        f"{args.root_dir}/{architecture}/784-1000-10,{ffunc}_{amp}_{phase},{lr}/*/")
    dir_list = sorted(dir_list)
    raw_data = []
    keys = []
    for _dir in dir_list:
        res = load_data(_dir)
        if res is None:
            continue
        label, result = res
        best_score = np.nanmax(result["acc_e"])
        params = parse.parse(
            "{architecture},{ffunc}_{amp}_{phase:d},{lr:e}", label)
        if params is None:
            continue
        keys = list(params.named.keys())
        values = list(params.named.values())
        # print(best_score.shape)
        raw_data.append(values + [float(best_score)])
        # print(_dir)
    df = pd.DataFrame(
        raw_data, columns=keys + ["best_score"])
    return df


def calc_correlation(x, y):
    return np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)


if __name__ == '__main__':
    fig = Figure(figsize=args.figsize)
    fig.create_grid((2, 2), hspace=0.05, height_ratios=(3, 1), width_ratios=(3, 1))

    df = load_performance("opto", architecture=args.architecture, amp="0.05", phase="30")
    print("{:.4f}±{:.4f} (best: {:.4f})".format(
        df["best_score"].mean(), df["best_score"].std(), df["best_score"].max()))

    dfs = []
    for amp in ["0.05", "0.1"]:
        df = load_performance(
            "opto", architecture=args.architecture,
            amp=amp, phase=args.phase)
        if len(df) == 0:
            continue
        ax = sns.lineplot(
            data=df, ax=fig[0, 0], x="phase", y="best_score", label=f"opto_{amp}",
            ci=args.ci, estimator=np.mean, legend="brief", marker="o", markersize=7.00)
        print(df.groupby("phase").mean())
        dfs.append(df)

    for idx, (arch, ffunc) in enumerate([("dfa", "constant"), ("dfa", "approx")]):
        df = load_performance(ffunc, architecture=arch)
        cmap = plt.get_cmap("tab10")
        print("{:.4f}±{:.4f} (best: {:.4f})".format(
            df["best_score"].mean(), df["best_score"].std(), df["best_score"].max()))
        fig[0, 0].line_y(
            df["best_score"].mean(), ls=":", lw=3.0,
            color=cmap(idx + 2), label=f"{ffunc}_{arch}")
        dfs.append(df)

    for idx, (arch, ffunc) in enumerate([("dfa", "acc_clip"), ("bp", "acc_clip")]):
        df = load_performance(ffunc, architecture=arch)
        fig[0, 0].line_y(
            df["best_score"].mean(), ls=":", lw=3.0,
            color=cmap(idx + 4), label=f"{ffunc}_{arch} (mean)")
        print("{:.4f}±{:.4f} (best: {:.4f})".format(
            df["best_score"].mean(), df["best_score"].std(), df["best_score"].max()))
        dfs.append(df)
        df = pd.concat(dfs)

    fig[0, 0].grid(True, which="both")
    fig[0, 0].set_xticklabels([])
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0)

    ts = np.linspace(-100, 100, 20001)
    # y_accurate = approx(0.0661 * ts)
    y_accurate = acc_clip(ts, clip_max=1.0)

    phases = np.arange(0, 181, 15)
    for amp in [0.05, 0.1]:
        corrs = []
        for phase in phases:
            y_opto = opto(ts, amp, phase)
            corrs.append(calc_correlation(y_opto, y_accurate))
        fig[1, 0].plot(phases, corrs, lw=1.0, marker="o", markersize=5.0)
        fig[1, 0].set_xlabel("phase")

    Figure.show(tight_layout=True)
    fig.savefig(f"../figure/{args.architecture}.png", dpi=300)
    fig.savefig(f"../figure/{args.architecture}.pdf", transparent=True)
    df.to_csv(f"../figure/{args.architecture}.csv")

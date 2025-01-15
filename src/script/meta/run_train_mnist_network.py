#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import string
import argparse
import platform
import itertools
import subprocess
import numpy as np

sys.path.append(".")

from pyutils.parallel import multi_process, for_each

parser = argparse.ArgumentParser()
parser.add_argument("--begin_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=None)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--node_num", type=int, default=None)

# experimental options
parser.add_argument('--root_dir', type=str, default='../result/train_mnist_network')

parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument("--net_dims", type=str, nargs="+", required=True)
parser.add_argument('--bfunc', type=str, nargs="+", required=True)
parser.add_argument("--lr", type=float, nargs="+", required=True)
parser.add_argument('--amp', type=float, default=1.0)
parser.add_argument('--phase', type=float, default=0.0)
parser.add_argument('--init_id', type=int, default=0)
parser.add_argument('--trial_num', type=int, default=5)

parser.add_argument('--final_only', action="store_true")
parser.add_argument('--use_bp', action="store_true")
parser.add_argument("--debug", action="store_true")
args, unknown = parser.parse_known_args()

bfunc_list = np.array(args.bfunc)
lr_list = np.array(args.lr)
exp_list = np.arange(args.trial_num) + args.init_id

if __name__ == '__main__':
    def _run(_id, _bfunc, _lr, _exp_id, worker_id=None):
        cmd = "python src/script/base/train_mnist_network.py "
        cmd += "--n_epochs {} ".format(args.n_epochs)
        cmd += "--net_dims {} ".format(" ".join(args.net_dims))
        cmd += "--bfunc {} ".format(_bfunc)
        cmd += "--amp {:g} ".format(args.amp)
        cmd += "--phase {:g} ".format(args.phase)
        cmd += "--lr {:.2e} ".format(_lr)

        if args.final_only:
            train_type = "fo"
            cmd += "--final_only "
        elif args.use_bp:
            train_type = "bp"
            cmd += "--use_bp "
        else:
            train_type = "dfa"
        root_dir = "{}/{}/{},{}_{:g}_{:g},{:.2e}/{}".format(
            args.root_dir, train_type, "-".join(args.net_dims),
            _bfunc, args.amp, args.phase, _lr, _exp_id)
        cmd += "--root_dir {} ".format(root_dir)
        cmd += " ".join(unknown) + " "
        print("[{}] : {}".format(_id, cmd))
        if not args.debug:
            subprocess.call(
                cmd.split(), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

    arg_list = list(itertools.product(bfunc_list, lr_list, exp_list))
    arg_list = [(_id,) + _ for _id, _ in enumerate(arg_list)]
    arg_list = arg_list[args.begin_id:args.end_id:args.split_num]

    if args.node_num is None:
        for_each(_run, arg_list, expand=True, verbose=False)
    else:
        multi_process(_run, arg_list, verbose=False, append_id=True,
                      expand=True, nodes=args.node_num)

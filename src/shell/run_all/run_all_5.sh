#!/bin/bash

python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test3/1.2 --target_radius 1.2 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test3/1.5 --target_radius 1.5 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test3/2 --target_radius 2 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test3/2.5 --target_radius 2.5 --bfunc opto


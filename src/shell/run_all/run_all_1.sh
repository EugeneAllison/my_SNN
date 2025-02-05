#!/bin/bash

python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test2/0.1 --target_radius 0.1 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test2/0.5 --target_radius 0.5 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test2/0.6 --target_radius 0.6 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test2/0.8 --target_radius 0.8 --bfunc opto


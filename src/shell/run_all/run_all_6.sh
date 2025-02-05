#!/bin/bash

python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test3/3 --target_radius 3 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_mnist_network_torch_B_sr.py --root_dir ./output/B_sr_amp04/test3/4 --target_radius 4 --bfunc opto


python /home/isi/li/my_SNN/src/script/base/train_Fmnist_network_torch_B_sr.py --root_dir ./output/B_sr_Fmnist/test2/0.1 --target_radius 0.1 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_Fmnist_network_torch_B_sr.py --root_dir ./output/B_sr_Fmnist/test2/0.5 --target_radius 0.5 --bfunc opto


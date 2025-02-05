#!/bin/bash

python /home/isi/li/my_SNN/src/script/base/train_Fmnist_network_torch_B_sr.py --root_dir ./output/B_sr_Fmnist/test2/0.6 --target_radius 0.6 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_Fmnist_network_torch_B_sr.py --root_dir ./output/B_sr_Fmnist/test2/0.8 --target_radius 0.8 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_Fmnist_network_torch_B_sr.py --root_dir ./output/B_sr_Fmnist/test2/1.0 --target_radius 1.0 --bfunc opto
python /home/isi/li/my_SNN/src/script/base/train_Fmnist_network_torch_B_sr.py --root_dir ./output/B_sr_Fmnist/test2/1.2 --target_radius 1.2 --bfunc opto

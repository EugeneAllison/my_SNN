#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torchvision import datasets, transforms


def get_data_loader(task_type, batch_size_train, batch_size_eval, subset_size=None):
    if task_type == 'mnist':
        data_train = datasets.MNIST(
            "../data/mnist", train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomAffine(
                    10, translate=[0.05, 0.05], shear=10,
                    scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]))
        data_eval = datasets.MNIST(
            "../data/mnist", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]))
        classes = list(range(10))
        image_size = (28, 28)

        def shape_func(image):
            return image.view(-1, 28, 28)
    elif task_type == 'fashion-mnist':
        data_train = datasets.FashionMNIST(
            "../data/fashion-mnist", train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomAffine(
                    10, translate=[0.05, 0.05], shear=10,
                    scale=(0.8, 1.2)),
                transforms.ToTensor(),
            ]))
        data_eval = datasets.FashionMNIST(
            "../data/fashion-mnist", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        classes = list(range(10))
        image_size = (28, 28)

        def shape_func(image):
            return image.view(-1, 28, 28)
    elif task_type == 'kmnist':
        data_train = datasets.KMNIST(
            "../data/kmnist", train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomAffine(
                    10, translate=[0.05, 0.05], shear=10,
                    scale=(0.8, 1.2)),
                transforms.ToTensor(),
            ]))
        data_eval = datasets.KMNIST(
            "../data/kmnist", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        classes = list(range(10))
        image_size = (28, 28)

        def shape_func(image):
            return image.view(-1, 28, 28)
    elif task_type == 'cifar10':
        data_train = datasets.CIFAR10(
            "../data/cifar10", train=True, download=True,
            transform=transforms.Compose([
                # transforms.RandomAffine(0, scale=(1.0, 1.1)),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     (0.4914, 0.4822, 0.4465),
                #     (0.2023, 0.1994, 0.2010)),
            ]))
        data_eval = datasets.CIFAR10(
            "../data/cifar10", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(
                #     (0.4914, 0.4822, 0.4465),
                #     (0.2023, 0.1994, 0.2010)),
            ]))
        classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
        image_size = (32, 96)

        def shape_func(image):
            return image.permute(0, 2, 3, 1).reshape(-1, 32, 96)

    if subset_size is None:
        shuffle = True
        sampler = None
    else:
        shuffle = False
        subset_indices = np.arange(int(subset_size))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(subset_indices)
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size_train, shuffle=shuffle, sampler=sampler)
    loader_eval = torch.utils.data.DataLoader(
        data_eval, batch_size=batch_size_eval, shuffle=False)
    return loader_train, loader_eval, classes, image_size, shape_func

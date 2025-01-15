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
import numpy as np
import pandas as pd
import re
from functools import reduce
from collections import defaultdict

SQUASH = [0.9595, 0.0661]
ALPHA = SQUASH[0] * SQUASH[1]
V_AVG = 8
V_SD = 10
A_AVG = ALPHA * V_AVG
A_SD = ALPHA * V_SD

class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        if method == 'VALID':

            self.eta = np.zeros((shape[0], int((shape[1] - ksize + 1) / self.stride), int((shape[1] - ksize + 1) / self.stride),
             self.output_channels))

        if method == 'SAME':
            self.eta = np.zeros((shape[0], int(shape[1]/self.stride), int(shape[2]/self.stride), self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (shape[1] - ksize) % stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - ksize) % stride != 0:
            print('input tensor height can\'t fit stride')


    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def gradient(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)



def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col

class AvgPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.integral = np.zeros(shape)
        self.index = np.zeros(shape)

    def gradient(self, eta):
        # stride = ksize
        next_eta = np.repeat(eta, self.stride, axis=1)
        next_eta = np.repeat(next_eta, self.stride, axis=2)
        next_eta = next_eta*self.index
        return next_eta/(self.ksize*self.ksize)

    def forward(self, x):
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(x.shape[1]):
                    row_sum = 0
                    for j in range(x.shape[2]):
                        row_sum += x[b, i, j, c]
                        if i == 0:
                            self.integral[b, i, j, c] = row_sum
                        else:
                            self.integral[b, i, j, c] = self.integral[b, i - 1, j, c] + row_sum

        out = np.zeros([x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride), self.output_channels],
                       dtype=float)

        # integral calculate pooling
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        self.index[b, i:i + self.ksize, j:j + self.ksize, c] = 1
                        if i == 0 and j == 0:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[
                                b, self.ksize - 1, self.ksize - 1, c]

                        elif i == 0:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[b, 1, j + self.ksize - 1, c] - \
                                                                          self.integral[b, 1, j - 1, c]
                        elif j == 0:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[b, i + self.ksize - 1, 1, c] - \
                                                                          self.integral[b, i - 1, 1, c]
                        else:
                            out[b, i / self.stride, j / self.stride, c] = self.integral[
                                                                              b, i + self.ksize - 1, j + self.ksize - 1, c] - \
                                                                          self.integral[
                                                                              b, i - 1, j + self.ksize - 1, c] - \
                                                                          self.integral[
                                                                              b, i + self.ksize - 1, j - 1, c] + \
                                                                          self.integral[b, i - 1, j - 1, c]

        out /= (self.ksize * self.ksize)
        return out




class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0], int(shape[1] / self.stride), int(shape[2] / self.stride), self.output_channels]

    def forward(self, x):

        out = np.zeros([x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride), self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, int(i / self.stride), int(j / self.stride), c] = np.max(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, int(i+index/self.stride), int(j + index % self.stride), c] = 1
        return out

    def gradient(self, eta):
        return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index


class Spikelayer(object):

    def __init__(self, shape, tau=20, h_th=0.4):
        shape = tuple(int(x) for x in shape)
        self.output_shape= shape
        self.tau = tau
        self.h_th = h_th
        self.eta = np.zeros(shape)
        self.v = np.zeros(shape)
        self.h = np.zeros(self.output_shape)
        self.refs = np.zeros(self.output_shape)
        self.a_current = np.zeros(self.output_shape)

    def init_state(self, keep_state=True):
        if hasattr(self, "v") and self.output_shape==self.v.shape and keep_state:
            return
        self.v = np.zeros(self.output_shape)
        self.h = np.zeros(self.output_shape)
        self.refs = np.zeros(self.output_shape)
        self.a_current = np.zeros(self.output_shape)

    def forward(self, inputs, dt=0.25):
        self.v = inputs
        self.refs[self.refs > 0] += dt
        self.refs[self.refs >= 1] = 0
        is_ref = self.refs > 0
        cv = dt / self.tau
        ch = 1 - cv
        self.h = (~is_ref) * (ch * self.h + cv * self.v )
        self.refs[np.logical_and(self.h > self.h_th, ~is_ref)] += 1e-8
        #current output
        self.a_current = np.logical_and(self.refs > 0, self.refs < 1.0)
        return self.a_current  #np.logical_and(self.refs > 0, self.refs < 1.0)

    def gradient(self, eta, bfunc):
        self.eta = eta
        g = bfunc(self.v)
        self.eta = self.eta * g
        return self.eta

    def DFA(self, bfunc):
        self.eta = bfunc(self.v)
        return self.eta

class Linearlayer(object):

    def __init__(self, shape, dim_out, connection=False,tau=20, h_th=0.4):
        self.input_shape = shape
        self.batchsize=shape[0]
        self.connection = connection
        if self.connection:
            self.dim_in = reduce(lambda x, y: x * y, shape[1:])
        else:
            self.dim_in = shape[1]

        self.dim_out = dim_out
        self.tau, self.h_th = tau, h_th
        self.output_shape = [self.batchsize, dim_out]

        self.W = np.ones((self.dim_in, self.dim_out))
        self.b = np.ones(self.dim_out)
        self.W_gradient = np.zeros(self.W.shape)
        self.b_gradient = np.zeros(self.b.shape)


    @staticmethod
    def __random_weight(N, v_avg=V_AVG, v_sd=V_SD, b_avg=0.8, alpha=ALPHA):
        v_sm = v_sd * v_sd + v_avg * v_avg
        # v2 = v_sigma + v1**2
        W_avg = (v_avg - b_avg) / (alpha * N * v_avg)
        # W1 = (v1 - b1) / (alpha * N * v1)
        W_sm = (v_sm + alpha**2 * (N - N**2) * W_avg**2 * v_avg**2 - 2 * alpha * N * b_avg * v_avg * W_avg - b_avg**2) / (alpha**2 * N * v_sm)
        # W2 = (v2 + alpha**2 * (N - N**2) * W1**2 * v1**2 - 2 * alpha * N * b1 * v1 * W1 - b1**2) / (alpha**2 * N * v2)
        W_sd = math.sqrt(W_sm - W_avg**2)
        # W_sigma = W2 - W1 ** 2
        return b_avg, W_avg, W_sd

    def init_weight(self, **kwargs):
        b_avg, W_avg, W_sd = self.__random_weight(self.dim_in)
        self.W = np.random.uniform(
            low=-math.sqrt(3), high=math.sqrt(3),
            size=(self.dim_in, self.dim_out)) * W_sd + W_avg

        self.b = np.ones(self.dim_out) * b_avg

        self._B = np.random.uniform(
            low=-math.sqrt(3), high=math.sqrt(3),
            size=(self.dim_out, self.dim_in)) * W_sd + W_avg
        self.scale = 2 / self.dim_in

    def forward(self, inputs):
        self.x = inputs.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.W) + self.b
        return output

    def DFA(self, e, eta):
        spk_eta = eta
        if hasattr(self, "B"):
            eta = np.dot(e, self.B) *eta
        else:
            eta = e * eta

        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.W_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.b.shape)

        if self.connection:
            next_eta = np.dot(spk_eta, self._B)
            next_eta = np.reshape(next_eta, self.input_shape)
            return next_eta

    def gradient(self, eta):

        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.W_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.b.shape)

        next_eta = np.dot(eta, self.W.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004, decay=True):
        # weight_decay = L2 regularization
        if decay:
            self.W *= (1 - weight_decay)
            self.b *= (1 - weight_decay)

        self.W -= alpha * self.W_gradient
        self.b -= alpha * self.b_gradient
        # zero gradient
        self.W_gradient = np.zeros(self.W.shape)
        self.b_gradient = np.zeros(self.b.shape)





class CSNNmodel(object):

    def __init__(self, dims, bf=0.0338, image_shape=[100, 28, 28, 1]):

        self.layers = []
        self.image_shape= image_shape
        dims = dims[0]


        class_num = int(dims[-2])
        linear_start = None

# expect dim: ['12C5', 'MP2','S', '64C5', 'MP2','S', 1000,'S' 10, 'S']

        for l in dims:
            print(type(l),l)
            if isinstance(l,str) and 'C' in l:
                pattern = r'(\d+)C(\d+)'
                matches = re.search(pattern, l)
                param = [int(match) for match in matches.groups()]
                if dims.index(l) == 0:
                    layer = Conv2D(image_shape, output_channels=param[0], ksize=param[1], stride=1)
                else:
                    shape = self.layers[-1].output_shape
                    layer = Conv2D(shape, output_channels=param[0],ksize=param[1],stride=1)
            elif isinstance(l,str) and 'S' in l:
                layer = Spikelayer(self.layers[-1].output_shape)
            elif isinstance(l,str) :
                pattern = r'[A-Za-z]+(\d+)'
                matches = re.search(pattern, l)
                param = [int(match) for match in matches.groups()]
                if l.startswith('MP'):
                    layer=MaxPooling(self.layers[-1].output_shape,ksize=param[0])
                else:
                    layer=AvgPooling(self.layers[-1].output_shape,ksize=param[0])
            else:
                if isinstance(dims[dims.index(l)-2],str):
                    layer=Linearlayer(self.layers[-1].output_shape, dim_out=l, connection=True)
                    linear_start = dims.index(l)
                    layer.init_weight()
                else:
                    layer=Linearlayer(self.layers[-1].output_shape, dim_out=l, connection=False)
                    layer.init_weight()

            self.layers.append(layer)

        B = np.eye(class_num)

        self.layers[-2].B = np.array(B)

        for l_pre, l_post in zip(
            reversed(self.layers[linear_start:-2:2]), reversed(self.layers[linear_start+2::2])
        ):


            B = np.dot(B, l_post._B)*bf
            l_pre.B = B


        for idx in range(linear_start, len(self.layers), 2):
            self.layers[idx].B *= self.layers[idx].scale


    def reset(self,  **kwargs):
        for l in self.layers:
            if isinstance(l, Spikelayer):
                l.init_state()


    def step(self, inputs, outputs, bfunc=None, lr=1.0, use_bp=False, decay=False ):
        a = inputs
        decay =decay
        for l in self.layers:
            a = l.forward(a)

        e = a - outputs
        # engage backward
        if bfunc is not None:
            if use_bp:
                eta=e
                for l in self.layers[::-1]:
                    if isinstance(l, Spikelayer):
                        eta = l.gradient(eta, bfunc)
                    elif isinstance(l, Linearlayer):
                        eta = l.gradient(eta*0.0338)
                        l.backward(alpha=lr, weight_decay=0.0004, decay=decay)
                    else:
                        eta = l.gradient(eta)
                        if isinstance(l, Conv2D):
                            l.backward(alpha=lr, weight_decay=0.0004)

             #use aDFA
            else:
                for l in self.layers[::-1]:
                    if isinstance(l, Spikelayer) and isinstance(self.layers[self.layers.index(l)-1], Linearlayer):
                        eta = l.DFA(bfunc)
                    elif isinstance(l, Linearlayer):
                        eta = l.DFA(e, eta)
                        l.backward(alpha=lr, weight_decay=0.0004, decay=decay)

                    elif isinstance(l, Spikelayer):
                        eta = l.gradient(eta,bfunc)

                    else:
                        eta = l.gradient(eta)
                        if isinstance(l, Conv2D):
                            l.backward(alpha=lr, weight_decay=0.0004)

        return a




















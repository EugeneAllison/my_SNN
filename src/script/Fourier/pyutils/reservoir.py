#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["DESN", "LESN", "Linear", "Softmax"]

import os
import sys
import joblib
import itertools
import scipy as sp
import numpy as np
import scipy.optimize
from numpy.random import RandomState
from pyutils.stats import sample_cross_correlation, optimize_ridge_criteria
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence


class ESN(object):
    def __init__(
            self, dim, g=1.0, noise_gain=None, activation=np.tanh,
            dtype=None, x_init=None, seed=None,
            scale=None, uniform=False, normalize=True,
            tunable=False, mu=1.0, alpha=1.0, select=None):
        self.dim, self.g = dim, g
        self.noise_gain = noise_gain
        self.f = activation

        self.dtype = dtype
        if x_init is None:
            self.x_init = np.zeros(dim, dtype=self.dtype)
        else:
            self.x_init = np.array(x_init, dtype=self.dtype)
            assert x_init.shape[-1] == dim, "shape of x_init must be" \
                " (..., {}) (matrix with shape {} was given.)".format(
                    self.dim, x_init.shape)
        self.x = np.array(self.x_init)
        self.rnd = RandomState(seed)

        # initializing internal values
        self.reset_weight(scale, uniform, normalize)

        # initialization for innate learning
        if tunable:
            self.reset_rls(mu, alpha, select)

    def reset_weight(self, scale=None, uniform=False, normalize=True):
        def _reset_weight():
            self.scale = 1.0 if scale is None else scale
            coeff = 1.0 / np.sqrt(self.dim * self.scale)
            self.w_net = self.rnd.randn(self.dim, self.dim) * coeff
            self.w_net = self.w_net.astype(self.dtype)
            self.uniform = uniform
            if uniform:
                w_con = np.full((self.dim, self.dim), False)
                w_con[:, :int(self.dim * self.scale)] = True
                for idx in range(self.dim):
                    self.rnd.shuffle(w_con[idx])
            else:
                w_con = np.full((self.dim * self.dim,), False)
                w_con[:int(self.dim * self.dim * self.scale)] = True
                self.rnd.shuffle(w_con)
                w_con = w_con.reshape((self.dim, self.dim))
            self.w_net = self.w_net * w_con
            if normalize and self.dim > 0:
                spectral_radius = max(abs(sp.sparse.linalg.eigs(
                    self.w_net, return_eigenvectors=False,
                    k=2, which="LM")))
                self.w_net = self.w_net / spectral_radius
        while True:
            try:
                _reset_weight()
                break
            except ArpackNoConvergence:
                continue

    def reset_rls(self, mu=1.0, alpha=1.0, select=None):
        self.tunable = True
        self.mu, self.alpha = mu, alpha
        w_pre_id = [self.w_net[idx].nonzero()[0] for idx in range(self.dim)]
        if select is None:
            self.w_pre = w_pre_id
        else:
            self.w_pre = [ids[select(ids)] for ids in w_pre_id]

        if self.uniform:
            size = self.w_pre[0].size
            self.P = np.zeros((self.dim, size, size))
            self.P[:] = np.eye(size, dtype=self.dtype) / self.alpha
        else:
            self.P = [
                np.eye(self.w_pre[idx].size, dtype=self.dtype) / self.alpha
                for idx in range(self.dim)]

    def to_pickle(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, mode="wb") as f:
            joblib.dump(self, f, compress=True)

    def f_g(self, x=None):
        if x is None:
            x = self.x
        return self.f(self.g * x)

    def innate(self, x_target, x_now=None, neuron_list=None, rho=None, is_global=False):
        def _innate(_now, _target):
            es = _target[neuron_list] - _now[neuron_list]
            if rho is None:
                qs = np.ones(es.shape[0])
            else:
                if is_global:
                    qs = np.exp(-rho * np.linalg.norm(es) ** 2)
                    qs = np.full(es.shape, qs)
                else:
                    qs = np.exp(-rho * es ** 2)
            mu_inv = 1 / self.mu
            if self.uniform:
                xs = np.array([_now[self.w_pre[idx]] for idx in neuron_list])
                Ps = self.P[neuron_list]
                ks = mu_inv * np.einsum("nij,nj->ni", Ps, xs)
                ls = np.einsum("ni,ni->n", xs, ks)
                gs = 1 / (1 + qs * ls)
                dPs = np.einsum("n,ni,nj->nij", gs, ks, ks)
                dws = np.einsum("n,n,ni->ni", gs, es, ks)
                self.P[neuron_list] = mu_inv * Ps - dPs
                for idx in neuron_list:
                    self.w_net[idx, self.w_pre[idx]] += dws[idx]
            else:
                for idx, e, q in zip(neuron_list, es, qs):
                    x = _now[self.w_pre[idx]]
                    k = mu_inv * self.P[idx].dot(x)
                    g = 1 / (1 + q * x.dot(k.T))
                    dP = g * np.outer(k, k)
                    self.P[idx] = mu_inv * self.P[idx] - dP
                    self.w_net[idx, self.w_pre[idx]] += k * (g * e)
            return es

        assert self.tunable, "option ***tunable*** must be set to True" \
            " when you call this function."
        if x_now is None:
            x_now = np.array(self.x)
        assert x_now.shape == x_target.shape, \
            "target shape must be same with that of current states (x_now)."
        if neuron_list is None:
            neuron_list = range(self.dim)

        if hasattr(self, "mode") and self.mode == "reverse":
            x_now = self.f_g(x_now)
            x_target = self.f_g(x_target)

        if x_now.ndim == 1:
            return _innate(x_now, x_target)
        elif val_now.ndim >= 2:
            errors = []
            for _now, _target in zip(
                    x_now.reshape(-1, self.dim),
                    x_target.reshape(-1, self.dim)):
                errors.append(_innate(_now, _target))
            return np.array(errors)

    def step(self):
        raise NotImplementedError

    def step_while(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def w_pre_id(self):
        if not hasattr(self, "_w_pre_id"):
            self._w_pre_id = [self.w_net[idx].nonzero()[0] for idx in range(self.dim)]
        return self._w_pre_id

    @staticmethod
    def read_pickle(file_name):
        with open(file_name, mode="rb") as f:
            net = joblib.load(f)
        return net


class DESN(ESN):
    '''
    Discrete-time echo state network

    Attributes:
        dim (int): number of nodes
        g (float): coefficient of function
        scale (float): density of connections
        noise_gain (float or array): Gaussian noise amplitude
        x_init (np.ndarray): init state
        activation: activation function of the network
        dtype (type): type of state and matrixs
        normalize (bool): normalize weight matrix if true
        seed (int): seed for random values (default: None)
        tunable (bool): enable tunable mode for innate training
        mu (float): forgetting parameter for innate training
        alpha (float): regularization strength for innate training
    '''
    def __init__(self, dim, **kwargs):
        super(DESN, self).__init__(dim, **kwargs)
        self.bias = None

    def step(self, u_in=None):
        self.x = self.x.dot(self.w_net.T)
        if u_in is not None:
            self.x += u_in
        if self.bias is not None:
            self.x += self.bias
        self.x = self.f_g(self.x)
        if self.noise_gain is not None:
            self.x += self.noise_gain * np.random.randn(self.dim)

    def step_while(self, num_step, u_in=None, init_step=0, verbose=False):
        for _ in range(init_step, num_step):
            if verbose:
                print("\x1b[2Kt={:.2f}".format(_), end="\r")
            if callable(u_in):
                self.step(u_in(_))
            else:
                self.step(u_in)

    def reset(self, x_reset=None):
        if x_reset is None:
            self.x = np.array(self.x_init)
        else:
            self.x = np.array(x_reset)

    def jacobian(self, x=None, use_cache=True):
        if x is None:
            x = self.x
        if not(use_cache and hasattr(self, "J1_")):
            self.J1_ = (self.w_net.T * self.g).T
        J2 = (1 / (np.cosh(self.g * x.dot(self.w_net.T))**2))
        return np.multiply(self.J1_.T, J2).T

    def maximum_lyapunov(self, num_step, u_in=None, num_trial=100,
                         perturbation=1e-6):
        result = []
        x_init = np.array(self.x)

        def _norm_rand():
            _noise = np.random.randn(num_trial, self.dim)
            return (_noise.T / np.linalg.norm(_noise, axis=1)).T
        x_pre = np.zeros((num_trial + 1, self.dim))
        x_pre[0] = self.x
        x_pre[1:] = self.x + perturbation * _norm_rand()
        self.reset(x_pre)
        self.step_while(num_step, u_in=u_in)
        x_post = np.array(self.x)
        d_post = np.linalg.norm(x_post[1:] - x_post[0], axis=1)
        result = np.log(d_post / perturbation) / num_step
        self.reset(x_init)  # reset to initial state
        return result


class LESN(ESN):
    '''
    Leaky echo state network

    Attributes:
        dim (int): number of ESN nodes
        tau (float): time constant of ESN
        g (float): coefficient of function
        mode (string): function types {"normal", "reverse"}
        scale (float): density of connection matrix
        noise_gain (float or array): Gaussian noise amplitude
        x_init (np.ndarray): init state
        activation: activation function of the network
        dtype (type): type of state and matrixs
        normalize (bool): normalize weight matrix if true
        seed (int): seed for random values (default: None)
        tunable (bool): enable tunable mode for innate training
        mu (float): forgetting parameter for innate training
        alpha (float): regularization strength for innate training
    '''
    def __init__(self, dim, tau, mode="normal", **kwargs):
        super().__init__(dim, **kwargs)
        self.tau = tau if np.isscalar(tau) else np.array(tau)
        assert mode in ["normal", "reverse"], \
            "option ***mode*** should be 'normal' or 'reverse'."
        self.mode = mode

    def fix_point(self, u_in=0.0, dim=None):
        net_range = slice(dim)
        x_init = np.zeros(self.dim)[net_range]
        _g = self.g if np.isscalar(self.g) else self.g[net_range]

        def _eq_normal(x, u_in=u_in):
            _x_in = _g * x.dot(
                self.w_net[net_range, net_range].T) + u_in[net_range]
            return -x[net_range] + self.f(_x_in)

        def _eq_reverse(x, u_in=u_in):
            _x_in = self.f(_g * x).dot(
                self.w_net[net_range, net_range].T)
            return -x + _x_in + u_in[net_range]

        if self.mode == "normal":
            return scipy.optimize.fsolve(_eq_normal, x_init)
        elif self.mode == "reverse":
            return scipy.optimize.fsolve(_eq_reverse, x_init)

    def step(self, dt, u_in=None):
        x_diff = np.zeros(self.x.shape)
        if u_in is None:
            u_in = 0.0
        if self.mode == "normal":
            x_diff += -self.x + self.f_g(self.x.dot(self.w_net.T) + u_in)
        elif self.mode == "reverse":
            x_diff += -self.x + self.f_g(self.x).dot(self.w_net.T) + u_in
        self.x += (dt / self.tau) * x_diff
        if self.noise_gain is not None:
            self.x += np.sqrt(2.0 * dt * self.noise_gain) * \
                np.random.randn(self.dim)

    def step_while(self, dt, T, u_in=None, t_init=0.0,
                   save=False, verbose=False):
        _t = t_init
        if save:
            record = np.zeros((int(T / dt), *net.x.shape))
        while _t < t_init + T:
            if save:
                record.append(self.x)
            if verbose:
                print("\x1b[2Kt={:.2f}".format(_t), end="\r")
            if callable(u_in):
                self.step(dt, u_in(_t))
            else:
                self.step(dt, u_in)
            _t += dt
        if save:
            return np.array(record)

    def reset(self, x_reset=None):
        if x_reset is None:
            self.x = np.array(self.x_init)
        else:
            self.x = np.array(x_reset)

    def jacobian(self, dt, x=None, use_cache=True):
        assert self.mode == "normal", "mode should be normal"
        if x is None:
            x = self.x
        if not(use_cache and hasattr(self, "J1_") and hasattr(self, "J2_")):
            self.J1_ = (1 - dt / self.tau) * np.eye(self.dim)
            self.J2_ = np.multiply(self.w_net.T, self.g * (dt / self.tau)).T
        return self.J1_ + self.J2_ * (1 / (np.cosh(self.g * x)**2))

    def maximum_lyapunov(self, dt, T, u_in=None, num_trial=100,
                         perturbation=1e-6, zero_range=None):
        result = []
        x_init = np.array(self.x)

        def _norm_rand():
            _noise = np.random.randn(num_trial, self.dim)
            if zero_range is not None:
                _noise[:, zero_range] = 0.0
            return (_noise.T / np.linalg.norm(_noise, axis=1)).T
        x_pre = np.zeros((num_trial + 1, self.dim))
        x_pre[0] = x_init
        x_pre[1:] = x_init + perturbation * _norm_rand()
        d_pre = np.linalg.norm(x_pre[1:] - x_pre[0], axis=1)
        self.reset(x_pre)
        self.step_while(dt, T, u_in=u_in)
        x_post = np.array(self.x)
        d_post = np.linalg.norm(x_post[1:] - x_post[0], axis=1)
        result = np.log(d_post / perturbation) / T
        self.reset(x_init)  # reset to previous state
        return result

    def maximum_lyapunov_dict(self, dt, T_list, u_in=None, num_trial=100,
                              perturbation=1e-6, zero_range=None):
        x_init = np.array(self.x)

        def _norm_rand():
            _noise = np.random.randn(num_trial, self.dim)
            if zero_range is not None:
                _noise[:, zero_range] = 0.0
            return (_noise.T / np.linalg.norm(_noise, axis=1)).T
        x_pre = np.zeros((num_trial + 1, self.dim))
        x_pre[0] = x_init
        x_pre[1:] = x_init + perturbation * _norm_rand()
        d_pre = np.linalg.norm(x_pre[1:] - x_pre[0], axis=1)

        result = {}
        self.reset(x_pre)
        T_pre = 0
        for _T in T_list:
            self.step_while(dt, _T - T_pre, u_in=u_in)
            x_post = np.array(self.x)
            d_post = np.linalg.norm(x_post[1:] - x_post[0], axis=1)
            result[_T] = np.log(d_post / perturbation) / _T
            T_pre = _T
        self.reset(x_init)  # reset to previous state
        return result

    def lyapunov_exponents(self, dt, x_list, size=None):
        r_list = []
        dim = x_list.shape[1]
        if size is None:
            size = self.dim
        q_pre = np.eye(self.dim, size)
        for _i, _x in enumerate(itertools.chain(x_list, x_list[::-1])):
            print("lyap: t={}".format(_i), end="\r")
            j = self.jacobian(dt, _x)
            q, r = np.linalg.qr(j.dot(q_pre))
            r_list.append(np.diag(r))
            q_pre = q
        l_list = np.log(np.abs(r_list))
        return l_list.mean(axis=0)

    @staticmethod
    def concatenate(net_list):
        dim = sum([_net.dim for _net in net_list])
        # concatenating tau
        tau = []
        for _net in net_list:
            _tau = _net.tau
            if np.isscalar(_tau):
                _tau = np.ones(_net.dim) * _net.tau
            tau.append(_tau)
        tau = np.concatenate(tau)
        # concatenating g
        g = []
        for _net in net_list:
            _g = _net.g
            if np.isscalar(_g):
                _g = np.ones(_net.dim) * _net.g
            g.append(_g)
        g = np.concatenate(g)
        # creating new concatenated network
        net = LESN(dim, tau, g=g)
        net.w_net = np.zeros((dim, dim))
        dim_offset = 0
        for _net in net_list:
            dim_term = dim_offset + _net.dim
            net.w_net[dim_offset:dim_term, dim_offset:dim_term] = _net.w_net
            dim_offset = dim_term
        return net


class Readout(object):
    def __init__(self, dim_in, dim_out, dtype=None, w_init=None,
                 seed=None, distribution="uniform", dist_args={},
                 mu=1.0, alpha=1.0):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dtype = dtype
        self.rnd = RandomState(seed)
        if w_init is not None:
            self.w_init = w_init
            self.w_init = self.w.reshape((self.dim_out, self.dim_in))
        else:
            if hasattr(self.rnd, distribution):
                self.w_init = getattr(self.rnd, distribution)(
                    size=(self.dim_out, self.dim_in), **dist_args)
            elif hasattr(np, distribution):
                self.w_init = getattr(np, distribution)(
                    (self.dim_out, self.dim_in), **dist_args)
        self.w_init = self.w_init.astype(self.dtype)
        self.bias = np.zeros(self.dim_out)
        self.alpha = alpha
        self.mu = mu
        self.mu_inv = 1 / mu
        self.reset()

    def __call__(self, x_list, **kwargs):
        return self.predict(x_list, **kwargs)

    def reset(self):
        self.P = []
        for _ in range(self.dim_out):
            self.P.append(
                np.eye(self.dim_in, dtype=self.dtype) * (1 / self.alpha))
        self.w = np.array(self.w_init)

    def ridge(self, X, Y, alpha=1e-8, fit_intercept=True, **kwargs):
        if alpha == "auto":
            alpha = optimize_ridge_criteria(X, Y, creteria="AIC")
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, **kwargs)
        model.fit(X, Y)
        self.w, self.bias = model.coef_, model.intercept_

    def force(self, x, d):
        assert x.shape == (self.dim_in,), "input shape should be (dim_in,)"
        e = d - self.predict(x)
        dws = np.zeros(self.dim_in, self.dim_out)
        for _i in range(self.dim_out):
            k = self.mu_inv * self.P[_i].dot(x)
            g = 1 / (1 + x.dot(k))
            dP = g * np.outer(k, k)
            dw = g * e[_i] * k
            self.P[_i] = self.mu_inv * self.P[_i] - dP
            self.w[_i] += dw
            dws[_i] = dw
        return dws

    def to_pickle(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, mode="wb") as f:
            joblib.dump(self, f, compress=True)

    @staticmethod
    def read_pickle(file_name):
        with open(file_name, mode="rb") as f:
            out = joblib.load(f)
        return out

    def predict(self, _input, **kwargs):
        raise NotImplementedError


class Linear(Readout):
    '''
    Linear tunable readout model

    Args:
        dim_in (int): input node dim.
        dim_out (int): output node dim.
        dtype (type, optional): numpy data type. Defaults to None.
        w_init (np.ndarray, optional): initial weight. Defaults to None.
        seed (int, optional): random seed. Defaults to None.
        distribution (str, optional): distribution type.
        dist_args (dict, optional): keyword args for np/rnd.
        mu (float, optional): forgetting parameter for innate training
        alpha (float, optional): regularization strength for innate training
    '''
    def predict(self, x):
        return np.array(x).dot(self.w.T) + self.bias


class Softmax(LogisticRegression):
    '''
    Alias of LogisticRegression
    '''
    pass

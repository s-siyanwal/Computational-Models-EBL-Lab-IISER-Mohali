"""Machine-learning tools for the *analysis* layer (never used inside the IBM).

M1  Gaussian-process denoiser: smooths noisy IBM outcomes over a parameter
    plane (heteroscedastic GP regression on per-cell estimates) to recover the
    emergent boundary of an IBM phenomenon without any oracle input.
M2  Neural emulator: an MLP trained only on IBM outcomes; afterwards compared
    with an analytic result (Kimura) on held-out conditions.
M3  Amortised inverse model: an MLP that maps IBM trajectories to the
    selection coefficients that generated them (simulation-based inference).
"""
from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def gp_denoise(X, y, y_var, X_new, length_scale=0.1, seed=0, min_length=None, var_floor=1e-3):
    """Heteroscedastic GP regression: per-point noise = sampling variance of
    the IBM estimate (y_var, floored) + a learned white-noise term.
    min_length: lower bound on the RBF length scale (default: the median grid
    spacing). Without it, cells with zero sampling variance (every replicate
    lost) force exact interpolation, the length scale collapses, and the
    posterior mean reverts to the prior mean between grid points.
    Returns (mean, sd, fitted gp) at X_new."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = y.mean()
    if min_length is None:
        gaps = [np.diff(np.unique(X[:, j])) for j in range(X.shape[1])]
        min_length = float(np.median(np.concatenate([g[g > 0] for g in gaps])))
    k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.full(X.shape[1], max(length_scale, min_length)),
                                               length_scale_bounds=(min_length, 10.0)) \
        + WhiteKernel(1e-3, (1e-8, 1.0))
    gp = GaussianProcessRegressor(kernel=k, alpha=np.maximum(np.asarray(y_var, float), var_floor),
                                  normalize_y=False, n_restarts_optimizer=3, random_state=seed)
    gp.fit(X, y - mu)
    m, sd = gp.predict(np.asarray(X_new, float), return_std=True)
    return m + mu, sd, gp


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def mlp(hidden=(64, 64), seed=0, max_iter=4000, alpha=1e-4):
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", alpha=alpha,
                                      max_iter=max_iter, early_stopping=True, n_iter_no_change=50,
                                      random_state=seed))


def logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))

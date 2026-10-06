"""Thin surrogate of the IBM (Layer 3). For experimental design and
sensitivity only: every claim about the papers is checked in the simulator.

GaussianProcessRegressor (default) with a HistGradientBoosting comparison;
inputs are standardised; leave-one-condition-out cross-validation is reported
so the surrogate is never trusted beyond its demonstrated accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def gp():
    k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
        + WhiteKernel(1e-2, (1e-6, 1.0))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=k, normalize_y=True,
                                                                    n_restarts_optimizer=3, random_state=0))


def gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)


def fit_and_validate(df, features, target, group_col):
    """Grouped CV (whole simulated conditions held out). Returns fitted GP,
    and a table of CV RMSE for GP and GBM vs the replicate-noise floor."""
    X = df[features].values
    y = df[target].values
    groups = df[group_col].values
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))
    out = {}
    for name, model in (("GP", gp()), ("GBM", gbm())):
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        out[name] = float(np.sqrt(np.mean((pred - y) ** 2)))
    # noise floor: within-condition replicate sd (no surrogate can beat it)
    out["replicate_sd"] = float(df.groupby(group_col)[target].std(ddof=1).mean())
    model = gp().fit(X, y)
    return model, out

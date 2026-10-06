"""Statistics for comparing the deterministic oracle with IBM ensembles."""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


def rmse(a, b, axis=None):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis)))


def loglog_slope(x, y):
    """OLS slope of log y on log x with 95% CI."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    res = stats.linregress(lx, ly)
    t = stats.t.ppf(0.975, len(lx) - 2)
    return res.slope, res.slope - t * res.stderr, res.slope + t * res.stderr


def agreement(pred_bool, prob, lo, hi):
    """Classify IBM proportions against a deterministic yes/no prediction:
    'agree' if CI excludes 0.5 on the predicted side, 'disagree' if it excludes
    0.5 on the other side, 'ambiguous' otherwise (near-neutral cells where
    drift dominates)."""
    pred_bool = np.asarray(pred_bool, bool)
    out = np.full(np.shape(prob), "ambiguous", dtype=object)
    out[(pred_bool & (lo > 0.5)) | (~pred_bool & (hi < 0.5))] = "agree"
    out[(pred_bool & (hi < 0.5)) | (~pred_bool & (lo > 0.5))] = "disagree"
    return out


def ks_vs_point(samples, point):
    """One-sample KS distance of the empirical endpoint distribution from a
    point mass (the deterministic prediction) -- equals the fraction of
    samples on the far side; reported together with mean absolute error."""
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    D = max(np.mean(s < point), np.mean(s > point))
    return dict(D=float(D), mae=float(np.mean(np.abs(s - point))))


def ks_two_sample(x, y):
    r = stats.ks_2samp(np.asarray(x)[np.isfinite(x)], np.asarray(y)[np.isfinite(y)])
    return dict(D=float(r.statistic), p=float(r.pvalue))


def fit_logistic(x, k, n):
    """Maximum-likelihood logistic fit P = 1/(1+exp(-(x-x50)/s)); returns x50, s."""
    from scipy.optimize import minimize
    x, k, n = (np.asarray(v, float) for v in (x, k, n))

    def nll(th):
        x50, ls = th
        p = 1 / (1 + np.exp(-(x - x50) / np.exp(ls)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))

    best = min((minimize(nll, [x0, np.log(0.05)], method="Nelder-Mead")
                for x0 in np.linspace(x.min(), x.max(), 5)), key=lambda r: r.fun)
    return float(best.x[0]), float(np.exp(best.x[1]))

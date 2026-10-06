"""Small statistics toolkit for the extensions (numpy/scipy only; the
statsmodels / lifelines packages are not available in the pinned environment).

  binom_two_sided / binom_min_tail   courts-first tests (Chinmay reported the
                                     smaller one-sided tail; spec uses two-sided)
  factorial_anova                    Type-III F tests, effect-coded factors with
                                     all interactions + additive block term
                                     (approximates JMP's mixed model with block
                                     random; balanced-ish designs)
  logrank                            k-sample log-rank test
  cox_wald                           Cox proportional hazards (Breslow ties),
                                     Wald chi-square per term
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import optimize, stats


def binom_two_sided(k, n):
    return float(stats.binomtest(int(k), int(n), 0.5).pvalue)


def binom_min_tail(k, n):
    """Smaller one-sided tail, which is what the thesis reports (verified on all
    16 rows of its Table 3.2)."""
    return float(min(stats.binom.cdf(k, n, 0.5), stats.binom.sf(k - 1, n, 0.5)))


def _effect_code(x):
    levels = sorted(set(x))
    if len(levels) != 2:
        raise ValueError("factorial_anova supports two-level factors")
    return np.where(np.asarray(x) == levels[1], 1.0, -1.0)


def factorial_anova(y, factors, block=None):
    """y: response; factors: {name: array of 2-level labels}; block: labels or
    None. Returns {term: (F, p, df1, df2)} for all main effects and
    interactions (Type III: drop one term's column(s) from the full model)."""
    y = np.asarray(y, float)
    names = list(factors)
    codes = {n: _effect_code(factors[n]) for n in names}
    cols, terms = [np.ones_like(y)], ["Intercept"]
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            c = np.ones_like(y)
            for n in combo:
                c = c * codes[n]
            cols.append(c)
            terms.append("*".join(combo))
    if block is not None:
        bl = np.asarray(block)
        levels = sorted(set(bl))
        for lv in levels[1:]:                       # sum-to-zero block coding
            cols.append(np.where(bl == lv, 1.0, 0.0) - np.where(bl == levels[0], 1.0, 0.0))
            terms.append("BLOCK")
    X = np.column_stack(cols)

    def rss(M):
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        r = y - M @ beta
        return float(r @ r)
    rss_full = rss(X)
    df2 = len(y) - np.linalg.matrix_rank(X)
    out = {}
    for t in dict.fromkeys(terms[1:]):
        keep = [i for i, tt in enumerate(terms) if tt != t]
        df1 = len(terms) - len(keep)
        F = ((rss(X[:, keep]) - rss_full) / df1) / (rss_full / df2)
        out[t] = (float(F), float(stats.f.sf(F, df1, df2)), df1, df2)
    return out


def logrank(time, event, group):
    """k-sample log-rank test. Returns (chi2, df, p)."""
    time, event, group = np.asarray(time, float), np.asarray(event, bool), np.asarray(group)
    gl = sorted(set(group))
    k = len(gl)
    O_E = np.zeros(k)
    V = np.zeros((k, k))
    for t in np.unique(time[event]):
        at_risk = time >= t
        n = at_risk.sum()
        d = (event & (time == t)).sum()
        if n < 2:
            continue
        ng = np.array([(at_risk & (group == g)).sum() for g in gl], float)
        dg = np.array([(event & (time == t) & (group == g)).sum() for g in gl], float)
        O_E += dg - d * ng / n
        V += (d * (n - d) / (n - 1)) * (np.diag(ng / n) - np.outer(ng, ng) / n ** 2)
    O_E, V = O_E[:-1], V[:-1, :-1]
    chi2 = float(O_E @ np.linalg.solve(V, O_E))
    return chi2, k - 1, float(stats.chi2.sf(chi2, k - 1))


def _cox_negll(beta, X, time, event, order):
    eta = X @ beta
    eta_o, ev_o, t_o = eta[order], event[order], time[order]
    # Breslow: risk set of each event time = all with time >= t (sorted descending)
    exp_eta = np.exp(eta_o - eta_o.max())
    cum = np.cumsum(exp_eta)                         # order is by decreasing time
    # for ties, use the cumulative sum at the last index with the same time
    last = np.searchsorted(-t_o, -t_o, side="right") - 1
    log_risk = np.log(cum[last]) + eta_o.max()
    return -float(np.sum((eta_o - log_risk)[ev_o]))


def cox_wald(time, event, factors):
    """Cox PH with effect-coded factors and all interactions. factors:
    {name: labels}; multi-level factors get k-1 sum-to-zero columns.
    Returns {term: (wald_chi2, df, p)}."""
    time, event = np.asarray(time, float), np.asarray(event, bool)
    names = list(factors)
    blocks = {}
    for n in names:
        lab = np.asarray(factors[n])
        lv = sorted(set(lab))
        blocks[n] = np.column_stack([np.where(lab == l, 1.0, 0.0) - np.where(lab == lv[-1], 1.0, 0.0)
                                     for l in lv[:-1]])
    cols, term_idx = [], {}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            mats = [blocks[n] for n in combo]
            prod = mats[0]
            for M in mats[1:]:
                prod = np.einsum("ni,nj->nij", prod, M).reshape(len(time), -1)
            term = "*".join(combo)
            term_idx[term] = list(range(sum(c.shape[1] for c in cols), sum(c.shape[1] for c in cols) + prod.shape[1]))
            cols.append(prod)
    X = np.column_stack(cols)
    order = np.argsort(-time, kind="stable")
    res = optimize.minimize(_cox_negll, np.zeros(X.shape[1]), args=(X, time, event, order), method="BFGS")
    beta = res.x
    # observed information by finite differences of the gradient
    eps = 1e-5
    p = len(beta)
    H = np.zeros((p, p))
    g0 = optimize.approx_fprime(beta, _cox_negll, eps, X, time, event, order)
    for i in range(p):
        b = beta.copy()
        b[i] += eps
        H[:, i] = (optimize.approx_fprime(b, _cox_negll, eps, X, time, event, order) - g0) / eps
    H = 0.5 * (H + H.T)
    cov = np.linalg.pinv(H)
    out = {}
    for term, idx in term_idx.items():
        b = beta[idx]
        W = float(b @ np.linalg.solve(cov[np.ix_(idx, idx)], b))
        out[term] = (W, len(idx), float(stats.chi2.sf(W, len(idx))))
    return out

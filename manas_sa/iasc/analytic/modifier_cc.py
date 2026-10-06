"""Exact re-implementation of the imperfect-modifier model of Singh, Jain,
Geeta Arun & Prasad (2023, bioRxiv 10.1101/2023.10.25.564090), following the
authors' own code in their Supplementary Information (PSAcommented.py for
autosomes, PSXcommented.py for X linkage):

* same parameter grids (including floating-point arange quirks and rounding),
* same fitness tables (Tables 1-2 incl. the absolute values in f22r, f23),
* same initial conditions (A2 at the Connallon-Clark equilibrium, clipped to
  [0.01, 0.99]; B2 = 0.05; D swept in steps of 0.01 within feasible bounds),
* same fixation rule: round(freq, 3) == 1 first reached within T = 3000
  generations, counted separately in females (x) and males (y),
* same early termination on negative frequencies or sums outside [0.98, 1.02).

Only the execution is compiled (numba) and parallelised over (k1, k2).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

T = 3000


def _py_round(v, nd):
    return float(round(float(v), nd))


def build_cases(xlinked):
    """Replicate the authors' nested loops; returns per-(k-independent) case
    arrays [h, sf, sm, D, rf, rm, A2f, A2m]."""
    SF = np.arange(0.2, 1.1, 0.1)
    Sf = [0.001] + [round(x, 1) for x in SF]
    HD = np.arange(0.1, 0.5, 0.1)
    H = [round(x, 1) for x in HD] + [0.49]
    RF = np.arange(0.1, 0.5, 0.1)
    Rf = [round(x, 1) for x in RF]
    Rm = Rf if not xlinked else [0.0]
    cases = []
    for h in H:
        for sf in Sf:
            if not xlinked:
                SM = np.arange(max(0, min(1, sf * h / (1 - h * (1 - sf)))),
                               max(0, min(1, sf * (1 - h) / h * (1 - sf))) + 0.00001, 0.1)
                Sm = [round(x, 2) for x in SM]
            else:
                SM = np.arange(max(0, min(1, 2 * sf * h / (1 + sf * h))),
                               max(0, min(1, 2 * sf * (1 - h) / (1 - sf * h))) + 0.00001, 0.1)
                Sm = [round(x, 1) for x in SM]
            for sm in Sm:
                if not xlinked:
                    A2f = max(0.01, min(0.99, (sm * (1 - h) - sf * h) / ((sm + sf) * (1 - 2 * h))))
                    A2m = A2f
                else:
                    A1f = max(0.01, min(0.99, (2 * sf * (1 - h) - sm) / (2 * sf * (1 - 2 * h))))
                    A2f = 1 - A1f
                    A2m = max(0.01, min(0.99, (sm - 2 * sf * h) / (2 * sf * (1 - 2 * h))))
                A1f, A1m = 1 - A2f, 1 - A2m
                B1, B2 = 0.95, 0.05
                x = [A1f * B1, A1f * B2, A2f * B1, A2f * B2]
                y = [A1m * B1, A1m * B2, A2m * B1, A2m * B2]
                possible = [0.25] + [v for z in x + y for v in (1 - z, z - 0)]
                Dr = np.arange(-min(possible), min(possible), 0.01)
                for D in Dr:
                    for rf in Rf:
                        for rm in Rm:
                            cases.append((h, sf, sm, D, rf, rm, A2f, A2m))
    return np.array(cases, dtype=np.float64)


@njit(cache=True)
def _round3_is_one(v):
    # Python round(v, 3) == 1  <=>  0.9995 <= v < 1.0005 (ties are measure-zero)
    return v >= 0.9995 and v < 1.0005


@njit(cache=True)
def _sum_ok(s):
    # round(s,3) in checklist where checklist = round(arange(0.98, 1.02, 0.001), 3)
    r = np.round(s, 3)
    return r >= 0.98 - 1e-9 and r <= 1.019 + 1e-9


@njit(cache=True)
def _run_case(k1, k2, h, sf, sm, D, rf, rm, A2f, A2m, xlinked, res):
    f11 = 1.0; f13 = 1 - k1 * sf; f31 = 1 - sf; f33 = 1 - (1 - k2) * sf
    f12 = 1 - k1 * h * sf; f32 = 1 - (1 + h - h * (1 - k2) - k2) * sf; f21 = 1 - h * sf
    f23 = 1 - abs((1 - k1 - k2) * h * sf); f22r = 1 - abs(h * sf * (1 - k1)); f22c = 1 - (1 - k2) * sf * h
    m11 = 1 - sm; m21 = 1 - h * sm; m33 = 1.0; m12 = 1 - sm; m13 = 1 - sm
    m22r = 1 - h * sm; m22c = 1 - h * sm; m23 = 1 - h * sm; m31 = 1.0; m32 = 1.0
    xm1 = 1 - sm; xm2 = 1 - sm; xm3 = 1.0; xm4 = 1.0                  # X-linked hemizygous males
    A1f = 1 - A2f; A1m = 1 - A2m
    x1 = A1f * 0.95 + D; x2 = A1f * 0.05 - D; x3 = A2f * 0.95 - D; x4 = A2f * 0.05 + D
    y1 = A1m * 0.95 + D; y2 = A1m * 0.05 - D; y3 = A2m * 0.95 - D; y4 = A2m * 0.05 + D
    # res: [A2f, B2f, A2B2f, A2m, B2m, A2B2m] fixation times (-1 = never)
    for q in range(6):
        res[q] = -1
    for t in range(T):
        wf = (x1 * y1 * f11 + (x1 * y2 + x2 * y1) * f12 + x2 * y2 * f13 + (x1 * y3 + x3 * y1) * f21
              + (x2 * y3 + x3 * y2) * f22r + (x1 * y4 + x4 * y1) * f22c + (x2 * y4 + x4 * y2) * f23
              + x3 * y3 * f31 + (x3 * y4 + x4 * y3) * f32 + x4 * y4 * f33)
        X1 = (2 * x1 * y1 * f11 + (x1 * y2 + x2 * y1) * f12 + (x1 * y3 + x3 * y1) * f21 + (1 - rf) * f22c * (x1 * y4 + x4 * y1) + rf * f22r * (x2 * y3 + x3 * y2)) / (2 * wf)
        X2 = (2 * x2 * y2 * f13 + (x1 * y2 + x2 * y1) * f12 + (x2 * y4 + x4 * y2) * f23 + (1 - rf) * f22r * (x3 * y2 + x2 * y3) + rf * f22c * (x4 * y1 + x1 * y4)) / (2 * wf)
        X3 = (2 * x3 * y3 * f31 + (x1 * y3 + x3 * y1) * f21 + (x3 * y4 + x4 * y3) * f32 + (1 - rf) * f22r * (x2 * y3 + x3 * y2) + rf * f22c * (x4 * y1 + x1 * y4)) / (2 * wf)
        X4 = (2 * x4 * y4 * f33 + (x2 * y4 + x4 * y2) * f23 + (x4 * y3 + x3 * y4) * f32 + (1 - rf) * f22c * (x4 * y1 + x1 * y4) + rf * f22r * (x2 * y3 + x3 * y2)) / (2 * wf)
        if not xlinked:
            wm = (x1 * y1 * m11 + (x1 * y2 + x2 * y1) * m12 + x2 * y2 * m13 + (x1 * y3 + x3 * y1) * m21
                  + (x2 * y3 + x3 * y2) * m22r + (x1 * y4 + x4 * y1) * m22c + (x2 * y4 + x4 * y2) * m23
                  + x3 * y3 * m31 + (x3 * y4 + x4 * y3) * m32 + x4 * y4 * m33)
            Y1 = (2 * x1 * y1 * m11 + (x1 * y2 + x2 * y1) * m12 + (x1 * y3 + x3 * y1) * m21 + (1 - rm) * m22c * (x1 * y4 + x4 * y1) + rm * m22r * (x2 * y3 + x3 * y2)) / (2 * wm)
            Y2 = (2 * x2 * y2 * m13 + (x1 * y2 + x2 * y1) * m12 + (x2 * y4 + x4 * y2) * m23 + (1 - rm) * m22r * (x3 * y2 + x2 * y3) + rm * m22c * (x4 * y1 + x1 * y4)) / (2 * wm)
            Y3 = (2 * x3 * y3 * m31 + (x1 * y3 + x3 * y1) * m21 + (x3 * y4 + x4 * y3) * m32 + (1 - rm) * m22r * (x2 * y3 + x3 * y2) + rm * m22c * (x4 * y1 + x1 * y4)) / (2 * wm)
            Y4 = (2 * x4 * y4 * m33 + (x2 * y4 + x4 * y2) * m23 + (x4 * y3 + x3 * y4) * m32 + (1 - rm) * m22c * (x4 * y1 + x1 * y4) + rm * m22r * (x2 * y3 + x3 * y2)) / (2 * wm)
        else:
            wm = x1 * xm1 + x2 * xm2 + x3 * xm3 + x4 * xm4
            Y1 = x1 * xm1 / wm; Y2 = x2 * xm2 / wm; Y3 = x3 * xm3 / wm; Y4 = x4 * xm4 / wm
        x1, x2, x3, x4 = X1, X2, X3, X4
        y1, y2, y3, y4 = Y1, Y2, Y3, Y4
        if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0:
            break
        if y1 < 0 or y2 < 0 or y3 < 0 or y4 < 0:
            break
        if _round3_is_one(x3 + x4) and res[0] < 0: res[0] = t
        if _round3_is_one(x2 + x4) and res[1] < 0: res[1] = t
        if _round3_is_one(x4) and res[2] < 0: res[2] = t
        if _round3_is_one(y3 + y4) and res[3] < 0: res[3] = t
        if _round3_is_one(y2 + y4) and res[4] < 0: res[4] = t
        if _round3_is_one(y4) and res[5] < 0: res[5] = t
        if not _sum_ok(x1 + x2 + x3 + x4):
            break
        if not _sum_ok(y1 + y2 + y3 + y4):
            break


@njit(parallel=True, cache=True)
def run_grid(kpairs, cases, xlinked):
    """Returns frac[nk, 6] (fraction of cases fixed) and mean time[nk, 6]
    (T + 25 when never fixed), exactly as the authors aggregate."""
    nk = kpairs.shape[0]
    nc = cases.shape[0]
    frac = np.zeros((nk, 6))
    tmean = np.zeros((nk, 6))
    for a in prange(nk):
        k1 = kpairs[a, 0]
        k2 = kpairs[a, 1]
        cnt = np.zeros(6)
        tsum = np.zeros(6)
        res = np.zeros(6, np.int64)
        for c in range(nc):
            _run_case(k1, k2, cases[c, 0], cases[c, 1], cases[c, 2], cases[c, 3], cases[c, 4], cases[c, 5],
                      cases[c, 6], cases[c, 7], xlinked, res)
            for q in range(6):
                if res[q] >= 0:
                    cnt[q] += 1
                    tsum[q] += res[q]
        for q in range(6):
            frac[a, q] = cnt[q] / nc
            tmean[a, q] = tsum[q] / cnt[q] if cnt[q] > 0 else T + 25
    return frac, tmean


OUTCOMES = ["A2 female", "B2 female", "A2B2 female", "A2 male", "B2 male", "A2B2 male"]

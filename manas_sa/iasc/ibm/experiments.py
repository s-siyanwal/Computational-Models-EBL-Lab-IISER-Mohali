"""In-silico versions of experiments that separate true sexual antagonism
(pleiotropy: the 'direct' term of COV(BV_f, BV_m)) from linkage-driven
negative covariance (Hill-Robertson interference / sex-limited selection with
LD: the 'indirect' term).

recombine_pool: k generations of FEMALE meiosis among a pool of hemigenomes,
    without selection or mutation (experimentally: hemiclone females crossed
    to each other, sons/daughters re-extracted through clone-generator
    females). LD decays by (1 - c) per generation; pleiotropy does not.
linkage_equilibrium: independent permutation of every site across the pool
    (the k -> infinity limit; keeps allele frequencies).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import engine as E


@njit(cache=True)
def _rounds(pool, k, WA, WX, pos, RA, RX, seed):
    np.random.seed(seed)
    n = pool.shape[0]
    cur = pool.copy()
    nxt = np.empty_like(pool)
    for _ in range(k):
        for i in range(n):
            a = np.random.randint(n)
            b = np.random.randint(n - 1)
            if b >= a:
                b += 1
            E._gamete(cur[a], cur[b], nxt[i], 0, WA, 0, pos, RA, True)
            if WX > 0:
                E._gamete(cur[a], cur[b], nxt[i], WA, WA + WX, WA * 64, pos, RX, True)
        tmp = cur
        cur = nxt
        nxt = tmp
    return cur


def recombine_pool(H, arch, k, R_A, R_X, seed):
    """H: (n, WA+WX) uint64 hemigenomes. R_A/R_X: map lengths PER REAL MEIOSIS
    (Morgans; not the lambda-rescaled values)."""
    if k == 0:
        return H.copy()
    pos = np.concatenate([arch.posA, arch.posX]) if arch.WX else arch.posA
    return _rounds(H, k, arch.WA, arch.WX, pos, float(R_A), float(R_X), int(seed))


def haplotype_values(H, arch):
    """Female and male values of each hemigenome (sum of per-copy effects;
    the male value counts the X once, as in a hemiclonal male)."""
    W = arch.WA + arch.WX
    bits = E.unpack(H, np.arange(len(H)), W).astype(np.float32)
    return bits @ arch.eff_f[: W * 64].astype(np.float32), bits @ arch.eff_m[: W * 64].astype(np.float32), bits


def linkage_equilibrium(bits, rng):
    return rng.permuted(bits, axis=0)

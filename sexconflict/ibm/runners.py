"""Convenience wrappers around the IBM kernels: founders -> ensemble -> measured
frequencies. Zero-leakage: no import from `sexconflict.deterministic`."""
from __future__ import annotations

import numpy as np

import os

from . import kernels as K


def backend():
    """'cpu' (numba, default) or 'opencl' (AMD/any GPU via pyopencl), from
    env SC_BACKEND; falls back to cpu if no OpenCL GPU is usable."""
    want = os.environ.get("SC_BACKEND", "cpu").lower()
    if want == "opencl":
        from . import opencl_backend as G
        if G.available():
            return G
    return K


def founders_haploid(N, pA1, pM1, rng):
    """Founding haplotype counts: round(pA1*N) individuals get A1 and
    round(pM1*N) get M1, the two assignments made independently at random
    (so linkage disequilibrium is ~0 by randomisation, not by formula)."""
    A = np.ones(N, np.int8)
    A[: int(round(pA1 * N))] = 0          # 0 = A1
    M = np.ones(N, np.int8)
    M[: int(round(pM1 * N))] = 0          # 0 = M1
    rng.shuffle(A)
    rng.shuffle(M)
    cls = 2 * A + M
    return np.bincount(cls, minlength=4).astype(np.int64)


def founders_from_haplotypes(N, x):
    """Founding counts proportional to given haplotype proportions (largest
    remainder rounding)."""
    x = np.asarray(x, float)
    raw = x / x.sum() * N
    c = np.floor(raw).astype(np.int64)
    rem = N - c.sum()
    c[np.argsort(-(raw - c))[:rem]] += 1
    return c


def run_haploid(init_counts, wtab, r, n_gen, n_rep, seed, stop_on_m_absorb=False):
    """init_counts: [4] (shared by all replicates) or [n_rep, 4]."""
    init = np.asarray(init_counts, np.int64)
    if init.ndim == 1:
        init = np.tile(init, (n_rep, 1))
    N = int(init[0].sum())
    assert np.all(init.sum(1) == N), "all replicates need the same N"
    counts = backend().haploid_ensemble(init, N, int(n_gen),
                                np.asarray(wtab, float), float(r), int(n_rep), int(seed),
                                bool(stop_on_m_absorb))
    x = counts / N
    return dict(x=x, pA1=x[..., 0] + x[..., 1], pM1=x[..., 0] + x[..., 2], N=N)


def run_diploid(N, init_A_copies, init_mito, wtab, n_gen, n_rep, seed):
    rec = backend().diploid_ensemble(int(init_A_copies), int(init_mito), int(N), int(n_gen),
                             np.asarray(wtab, float), int(n_rep), int(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(
            p_zyg=rec[..., 0] / (2 * N),           # A freq among zygotes
            mito=rec[..., 1] / N,                  # cytoplasm-1 freq among zygotes
            p_f=rec[..., 2] / (2 * rec[..., 3]),   # A freq in surviving females
            p_m=rec[..., 4] / (2 * rec[..., 5]),   # A freq in surviving males
            N=N,
        )


def run_xlinked(N, init_copies, wtab, n_gen, n_rep, seed):
    rec = K.xlinked_ensemble(int(init_copies), int(N), int(n_gen),
                             np.asarray(wtab, float), int(n_rep), int(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(p_f=rec[..., 0] / (2 * rec[..., 1]), p_m=rec[..., 2] / rec[..., 3], N=N)


def ibm_iasc_burn_in(N, wtab, n_rep, seed, n_gen=600):
    """Let each replicate population find its own IaSC state: start at 50% A1,
    no modifier, and run the life cycle. Returns the A1 count per replicate.
    (The IBM never sees the analytical p*.)"""
    init = np.array([0, N // 2, 0, N - N // 2], np.int64)
    out = run_haploid(init, wtab, 0.5, n_gen, n_rep, seed)
    return np.rint(out["pA1"][:, -1] * N).astype(np.int64)


def introduce_modifier(nA1, N, m1_copies, rng):
    """Mutate m1_copies randomly chosen individuals from M2 to M1.
    Returns founder counts [n_rep, 4]."""
    nA1 = np.atleast_1d(nA1)
    out = np.empty((nA1.size, 4), np.int64)
    for i, na in enumerate(nA1):
        on_A1 = rng.hypergeometric(na, N - na, m1_copies)
        out[i] = [on_A1, na - on_A1, m1_copies - on_A1, N - na - (m1_copies - on_A1)]
    return out


def run_xy(N, pX0, pY0, wtab, r, n_gen, n_rep, seed):
    """SA locus linked (recombination r) to an XY sex-determining region.
    wtab[sex, #A1] survival weights (sex 0 = XX female, 1 = XY male)."""
    rec = K.xy_ensemble(float(pX0), float(pY0), int(N), int(n_gen), np.asarray(wtab, float), float(r),
                        int(n_rep), int(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(pX=rec[..., 0] / rec[..., 1], pY=rec[..., 2] / rec[..., 3], N=N)

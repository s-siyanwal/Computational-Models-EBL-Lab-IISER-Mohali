"""Population-level statistics, computed the way Geeta Arun (2025) computes
them: from a sample of individuals' genotypes, r_mf,W = a_f' L a_m /
sqrt(a_f' L a_f * a_m' L a_m), with L the covariance matrix of genotype
states. Because a' L b = Cov(sum_i a_i g_i, sum_j b_j g_j), this equals the
correlation between female and male breeding values over sampled genotypes.
Direct and indirect (LD-mediated) parts are separated using per-site
variances (Eq. 1 of the preprint)."""
from __future__ import annotations

import numpy as np

from . import engine as E


def population_stats(genomes, arch, rng, n_f=None, n_m=None, autosome_only=True):
    N2 = genomes.shape[0]
    N = N2 // 2
    nf_pop = N // 2
    n_f = nf_pop if n_f is None else min(n_f, nf_pop)
    n_m = N - nf_pop if n_m is None else min(n_m, N - nf_pop)
    idx = np.concatenate([rng.choice(nf_pop, n_f, replace=False),
                          nf_pop + rng.choice(N - nf_pop, n_m, replace=False)])
    WA = arch.WA
    W = WA if autosome_only else arch.WA + arch.WX
    rows = np.empty(2 * len(idx), np.int64)
    rows[0::2] = 2 * idx
    rows[1::2] = 2 * idx + 1
    bits = E.unpack(genomes, rows, W).astype(np.float32)
    g = 0.5 * (bits[0::2] + bits[1::2])                     # genotype state 0 / .5 / 1
    ef = arch.eff_f[: W * 64]
    em = arch.eff_m[: W * 64]
    # average effects on the genotype-state scale (state 0 -> 1 = 2 copies)
    af, am = 2 * ef, 2 * em
    bvf = g @ af
    bvm = g @ am
    Vf = bvf.var(ddof=1)
    Vm = bvm.var(ddof=1)
    C = np.cov(bvf, bvm)[0, 1]
    vg = g.var(0, ddof=1)
    direct = float(np.sum(af * am * vg))
    p = bits.mean(0)
    seg = int(((p > 0) & (p < 1)).sum())
    r = C / np.sqrt(Vf * Vm) if Vf > 0 and Vm > 0 else np.nan
    return dict(r_mf=float(r), V_f=float(Vf), V_m=float(Vm), COV=float(C), direct=direct,
                indirect=float(C - direct), interference_pct=float(100 * C / Vf) if Vf > 0 else np.nan,
                n_seg=seg, n_sampled=len(idx))


def ld_by_distance(genomes, arch, R_map, rng, bins=(0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0),
                   max_pairs=200000, kinds=(1, 2)):
    """Signed LD between derived alleles at a female-limited (kind 1) and a
    male-limited (kind 2) segregating site, binned by map distance (Morgans,
    female map). D(derived, derived) has the same sign as D(beneficial,
    beneficial), so negative values are HRI-type repulsion."""
    N2 = genomes.shape[0]
    bits = E.unpack(genomes, np.arange(N2), arch.WA).astype(np.float32)
    p = bits.mean(0)
    seg = np.where((p > 0) & (p < 1))[0]
    k = arch.kind[seg]
    fi = seg[k == kinds[0]]
    mi = seg[k == kinds[1]]
    if len(fi) == 0 or len(mi) == 0:
        return None
    n = min(max_pairs, len(fi) * len(mi))
    i = fi[rng.integers(0, len(fi), n)]
    j = mi[rng.integers(0, len(mi), n)]
    D = (bits[:, i] * bits[:, j]).mean(0) - p[i] * p[j]
    w = (2 * arch.eff_f[i]) * (2 * arch.eff_m[j])          # contribution weight to COV
    dist = np.abs(arch.posA[i] - arch.posA[j]) * R_map
    b = np.digitize(dist, bins) - 1
    out = []
    for q in range(len(bins) - 1):
        m = b == q
        if m.sum() < 20:
            continue
        out.append(dict(bin_lo=bins[q], bin_hi=bins[q + 1], mid=float(np.median(dist[m])), n_pairs=int(m.sum()),
                        mean_D=float(D[m].mean()), se_D=float(D[m].std() / np.sqrt(m.sum())),
                        mean_contrib=float((w[m] * D[m]).mean())))
    return out

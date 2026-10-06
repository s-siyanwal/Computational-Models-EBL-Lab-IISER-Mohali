"""Deterministic two-sex, two-locus recursion (diploid; autosomal or X-linked),
generalising Connallon & Clark (2010) to arbitrary sex-specific genotype
fitness tables and sex-specific recombination (r_m = 0 for Drosophila males).

Haplotype order (locus A first): [A1B1, A1B2, A2B1, A2B2]; index = 2*a + b.
State: x = egg (female-gamete) haplotype frequencies, y = sperm haplotype
frequencies (X-bearing sperm in the X-linked case).

Autosomal life cycle: zygote = (egg i, sperm j) with prob x_i y_j -> sex 50:50
-> viability/fertility weight F[i,j] (females) or M[i,j] (males) -> meiosis
with crossover prob r_f / r_m -> gametes -> optional mutation.
X-linked: daughters = (egg i, X-sperm j) weighted by F[i,j]; sons carry the
egg haplotype i, weighted by M1[i]; X-sperm = son's X (no male recombination).

Everything broadcasts over a leading batch dimension so whole parameter grids
are iterated at once.
"""
from __future__ import annotations

import numpy as np

A_OF = np.array([0, 0, 1, 1])
B_OF = np.array([0, 1, 0, 1])


def _recomb_tensor(r):
    """G[..., i, j, k] = P(gamete k | parental haplotypes i, j), crossover prob r."""
    r = np.asarray(r, float)
    G = np.zeros(r.shape + (4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k, (a_src, b_src) in enumerate(((i, i), (j, j), (i, j), (j, i))):
                w = (1 - r) / 2 if k < 2 else r / 2
                hap = 2 * A_OF[a_src] + B_OF[b_src]
                G[..., i, j, hap] += w
    return G


def step_autosomal(x, y, F, M, r_f, r_m, mu=None):
    """x, y: [..., 4]; F, M: [..., 4, 4] (symmetric); r_f, r_m: [...]."""
    Gf, Gm = _recomb_tensor(r_f), _recomb_tensor(r_m)
    Z = x[..., :, None] * y[..., None, :]                       # zygote (egg i, sperm j)
    Zs = 0.5 * (Z + np.swapaxes(Z, -1, -2))                     # unordered genotype weights
    wf = Zs * F
    wm = Zs * M
    xn = np.einsum("...ij,...ijk->...k", wf, Gf) / wf.sum((-1, -2))[..., None]
    yn = np.einsum("...ij,...ijk->...k", wm, Gm) / wm.sum((-1, -2))[..., None]
    if mu is not None:
        xn, yn = mutate(xn, mu), mutate(yn, mu)
    return xn, yn


def step_xlinked(x, y, F, M1, r_f, mu=None):
    """F: [...,4,4] daughters; M1: [...,4] hemizygous sons (haplotype = egg)."""
    Gf = _recomb_tensor(r_f)
    Z = x[..., :, None] * y[..., None, :]
    Zs = 0.5 * (Z + np.swapaxes(Z, -1, -2))
    wf = Zs * F
    xn = np.einsum("...ij,...ijk->...k", wf, Gf) / wf.sum((-1, -2))[..., None]
    ws = x * M1
    yn = ws / ws.sum(-1, keepdims=True)
    if mu is not None:
        xn, yn = mutate(xn, mu), mutate(yn, mu)
    return xn, yn


def mutate(h, mu):
    """Forward mutation 1 -> 2 at each locus with rate mu = (mu_A, mu_B)."""
    muA, muB = mu
    a1 = h[..., 0] + h[..., 1]
    out = h.copy()
    # locus A: A1 -> A2
    out = out.copy()
    tA = np.stack([h[..., 0] * muA, h[..., 1] * muA], -1)
    out[..., 0] -= tA[..., 0]
    out[..., 1] -= tA[..., 1]
    out[..., 2] += tA[..., 0]
    out[..., 3] += tA[..., 1]
    tB = np.stack([out[..., 0] * muB, out[..., 2] * muB], -1)
    out[..., 0] -= tB[..., 0]
    out[..., 2] -= tB[..., 1]
    out[..., 1] += tB[..., 0]
    out[..., 3] += tB[..., 1]
    return out


def ld(h):
    """D = x11 x22 - x12 x21 for haplotype frequencies h[..., 4]."""
    return h[..., 0] * h[..., 3] - h[..., 1] * h[..., 2]


def allele1(h):
    return h[..., 0] + h[..., 1], h[..., 0] + h[..., 2]


# --------------------------------------------------------------------------
# Fitness tables
# --------------------------------------------------------------------------
def additive_two_locus(eA, eB, h=0.5):
    """Genotype fitness 1 + sum of additive (dominance h) effects of allele 2
    at each locus: eA, eB are the per-sex homozygous effects (negative =
    deleterious). Returns [..., 4, 4] table."""
    eA, eB = np.asarray(eA, float), np.asarray(eB, float)
    T = np.zeros(np.broadcast(eA, eB).shape + (4, 4))
    for i in range(4):
        for j in range(4):
            nA = A_OF[i] + A_OF[j]
            nB = B_OF[i] + B_OF[j]
            dA = {0: 0.0, 1: h, 2: 1.0}[nA]
            dB = {0: 0.0, 1: h, 2: 1.0}[nB]
            T[..., i, j] = 1 + eA * dA + eB * dB
    return T


def sex_limited_tables(s_f, s_m, h=0.5):
    """Locus A female-limited (allele 2 deleterious, s_f), locus B male-limited
    (allele 2 deleterious, s_m). No SA selection anywhere."""
    F = additive_two_locus(-np.asarray(s_f, float), 0.0 * np.asarray(s_f, float), h)
    M = additive_two_locus(0.0 * np.asarray(s_m, float), -np.asarray(s_m, float), h)
    return F, M


def sa_single_locus_tables(s_f, s_m, h=0.5):
    """Locus A sexually antagonistic (allele 2 good for males, bad for
    females); locus B neutral. Used for the 'true SA' regime."""
    s_f, s_m = np.asarray(s_f, float), np.asarray(s_m, float)
    F = additive_two_locus(-s_f, 0 * s_f, h)
    M = additive_two_locus(+s_m, 0 * s_m, h)
    return F, M


# --------------------------------------------------------------------------
# Deterministic 'sex-differential' LD (derived here; claim type C)
# --------------------------------------------------------------------------
def sex_differential_ld_one_step(dA, dB, r):
    """LD created in one round of gamete formation from pooled egg/sperm
    haplotypes that are each at linkage equilibrium but differ in allele
    frequency by dA (egg - sperm, locus A) and dB (locus B):
        D' = (1 - 2r) dA dB / 4
    Negative between female-beneficial and male-beneficial alleles when
    selection is purely sex-limited (dA > 0, dB < 0 for the beneficial alleles)."""
    return (1 - 2 * np.asarray(r)) * dA * dB / 4

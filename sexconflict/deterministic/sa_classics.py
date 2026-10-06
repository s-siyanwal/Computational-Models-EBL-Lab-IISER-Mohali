"""Classical sexually-antagonistic (SA) results used as extra test cases.

Fitness scheme (Kidwell et al. 1977; Rice 1984; Fry 2010, Table 1).
A1 = male-beneficial allele, A2 = female-beneficial allele:

              A1A1        A1A2           A2A2
   males      1           1 - h_m s_m    1 - s_m
   females    1 - s_f     1 - h_f s_f    1
   (X-linked males: A1 -> 1, A2 -> 1 - s_m)

Protected polymorphism = each allele increases when rare. The conditions are
Parsons' (1961) latent root applied at both boundaries. Written out (Fry 2010):
   autosomal (eq. 3): h_f/(1-h_m+h_f s_f) < s_m/s_f < (1-h_f)/(h_m(1-s_f))
   additive  (eq. 1): 1/(1+s_f) < s_m/s_f < 1/(1-s_f)       [Kidwell wedge]
   X-linked  (eq. 2): 2h_f/(1+h_f s_f) < s_m/s_f < 2(1-h_f)/(1-h_f s_f)

Also here:
 * Kimura's (1962) diffusion fixation probability with sex-averaged selection,
   and its maternal-cytoplasm analogue (female selection only, Ne = N_f);
 * a two-locus SA x sex-determining-region recursion (Rice 1987): A1 on X in
   eggs / X-sperm / Y-sperm, recombination r between the SA locus and the SDR;
 * the diffusion mean time to loss of a balanced polymorphism (backward
   Kolmogorov equation), used against the IBM's persistence times.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from . import owen_parsons as op


# --------------------------------------------------------------- fitness
def owen_params(s_f, s_m, h_f, h_m):
    """(a1,h1,b1,a2,h2,b2) for owen_parsons with the counted allele 'A' = A1."""
    return (1 - s_f, 1 - h_f * s_f, 1.0, 1.0, 1 - h_m * s_m, 1 - s_m)


def xlinked_params(s_f, s_m, h_f):
    """(a1,h1,b1,a2,b2) for owen_parsons.sexlinked_* with 'A' = A1."""
    return (1 - s_f, 1 - h_f * s_f, 1.0, 1.0, 1 - s_m)


# ---------------------------------------------------- invasion conditions
def autosomal_A1_invades(s_f, s_m, h_f, h_m):
    a1, h1, b1, a2, h2, b2 = owen_params(s_f, s_m, h_f, h_m)
    return op.autosomal_invades(h1, b1, h2, b2)


def autosomal_A2_invades(s_f, s_m, h_f, h_m):
    a1, h1, b1, a2, h2, b2 = owen_params(s_f, s_m, h_f, h_m)
    # relabel: A2 rare, resident A1A1 -> 'b' viabilities are a1, a2
    return op.autosomal_invades(h1, a1, h2, a2)


def autosomal_protected(s_f, s_m, h_f, h_m):
    return autosomal_A1_invades(s_f, s_m, h_f, h_m) & autosomal_A2_invades(s_f, s_m, h_f, h_m)


def fry_eq3(s_f, s_m, h_f, h_m):
    s_f, s_m = np.asarray(s_f, float), np.asarray(s_m, float)
    ratio = s_m / s_f
    return (ratio > h_f / (1 - h_m + h_f * s_f)) & (ratio < (1 - h_f) / (h_m * (1 - s_f)))


def xlinked_A1_invades(s_f, s_m, h_f):
    a1, h1, b1, a2, b2 = xlinked_params(s_f, s_m, h_f)
    return op.sexlinked_invades(h1, b1, a2, b2)


def xlinked_A2_invades(s_f, s_m, h_f):
    a1, h1, b1, a2, b2 = xlinked_params(s_f, s_m, h_f)
    return op.sexlinked_invades(h1, a1, b2, a2)


def xlinked_protected(s_f, s_m, h_f):
    return xlinked_A1_invades(s_f, s_m, h_f) & xlinked_A2_invades(s_f, s_m, h_f)


def rice_eq2(s_f, s_m, h_f):
    s_f, s_m = np.asarray(s_f, float), np.asarray(s_m, float)
    ratio = s_m / s_f
    return (ratio > 2 * h_f / (1 + h_f * s_f)) & (ratio < 2 * (1 - h_f) / (1 - h_f * s_f))


# --------------------------------------------------------------- Kimura
def kimura_u(p, N, s):
    """Fixation probability of an allele at frequency p with genic selection s
    per copy (fitnesses 1, 1+s, 1+2s), diploid Ne = N: (1-e^{-4Nsp})/(1-e^{-4Ns})."""
    p, N, s = np.broadcast_arrays(*(np.asarray(v, float) for v in (p, N, s)))
    x = 4 * N * s
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u = -np.expm1(-x * p) / -np.expm1(-x)
    return np.where(np.abs(x) < 1e-9, p, u)


def kimura_u_haploid(p, M, s):
    """Haploid Wright-Fisher (M copies, e.g. mtDNA through M = N_f mothers):
    (1-e^{-2Msp})/(1-e^{-2Ms})."""
    p, M, s = np.broadcast_arrays(*(np.asarray(v, float) for v in (p, M, s)))
    x = 2 * M * s
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u = -np.expm1(-x * p) / -np.expm1(-x)
    return np.where(np.abs(x) < 1e-9, p, u)


# ------------------------------------------- SA locus linked to the SDR
def xy_step(pXe, pXs, pY, s_f, s_m, h_f, h_m, r):
    """One generation of the SA x sex-determining-region model.
    pXe: A1 freq on X in eggs; pXs: on X-bearing sperm; pY: on Y-bearing sperm.
    Daughters = egg x X-sperm, sons = egg x Y-sperm; viability selection; then
    females transmit an X (allele freq among surviving daughters); males
    transmit X or Y, and the SA allele travels with the transmitted sex
    chromosome with prob 1-r (crossover with the SDR with prob r)."""
    wf = (1.0, 1 - h_f * s_f, 1 - s_f)      # #A1 = 0 -> A2A2 (1); 1 -> het; 2 -> A1A1 (1-s_f)
    wm = (1 - s_m, 1 - h_m * s_m, 1.0)      # #A1 = 0 -> A2A2 (1-s_m); 1 -> het; 2 -> A1A1 (1)
    # daughters
    d0, d1, d2 = (1 - pXe) * (1 - pXs), pXe * (1 - pXs) + (1 - pXe) * pXs, pXe * pXs
    Wd = d0 * wf[0] + d1 * wf[1] + d2 * wf[2]
    pXe_n = (0.5 * d1 * wf[1] + d2 * wf[2]) / Wd
    # sons: X from egg (x), Y from sperm (y); joint independent
    s00, s10, s01, s11 = (1 - pXe) * (1 - pY), pXe * (1 - pY), (1 - pXe) * pY, pXe * pY
    w00, w10, w01, w11 = wm[0], wm[1], wm[1], wm[2]
    Ws = s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11
    x_bar = (s10 * w10 + s11 * w11) / Ws       # A1 on the X of surviving sons
    y_bar = (s01 * w01 + s11 * w11) / Ws       # A1 on the Y of surviving sons
    pXs_n = (1 - r) * x_bar + r * y_bar
    pY_n = (1 - r) * y_bar + r * x_bar
    return pXe_n, pXs_n, pY_n


def xy_iterate(p0, params, n_gen=20000, tol=1e-13):
    pXe, pXs, pY = (np.asarray(v, float) for v in p0)
    for _ in range(n_gen):
        n = xy_step(pXe, pXs, pY, *params)
        d = max(np.max(np.abs(n[0] - pXe)), np.max(np.abs(n[1] - pXs)), np.max(np.abs(n[2] - pY)))
        pXe, pXs, pY = n
        if d < tol:
            break
    return pXe, pXs, pY


def xy_polymorphic(s_f, s_m, h_f, h_m, r, n_gen=20000):
    """Both alleles maintained (from an interior start) -- the protected
    region of the linked model. Returns (polymorphic, pXe, pY) at equilibrium."""
    pXe, pXs, pY = xy_iterate((0.5, 0.5, 0.5), (s_f, s_m, h_f, h_m, r), n_gen=n_gen)
    total = (2 * pXe + pXs + pY) / 4  # rough allele freq across chromosome classes
    return (total > 1e-6) & (total < 1 - 1e-6), pXe, pY


# ------------------------------------------- diffusion: time to loss
def mean_absorption_time(M, V, p0, n=4001):
    """Solve 0.5 V(p) T'' + M(p) T' = -1, T(0) = T(1) = 0 (backward
    Kolmogorov equation for the mean time to loss/fixation) on a uniform grid
    by central finite differences; returns T(p0)."""
    x = np.linspace(0, 1, n)
    h = x[1] - x[0]
    xi = x[1:-1]
    m, v = M(xi), V(xi)
    lower = 0.5 * v / h ** 2 - m / (2 * h)
    diag = -v / h ** 2
    upper = 0.5 * v / h ** 2 + m / (2 * h)
    ab = np.zeros((3, n - 2))
    ab[0, 1:] = upper[:-1]
    ab[1] = diag
    ab[2, :-1] = lower[1:]
    T = solve_banded((1, 1), ab, -np.ones(n - 2))
    return np.interp(p0, xi, T)


def fixation_probability(M, V, p0, n=20001):
    """Diffusion fixation probability with an arbitrary drift M(p) and
    variance V(p) (Kimura 1962, general form):
        u(p0) = int_0^p0 psi / int_0^1 psi,   psi(y) = exp(-int_0^y 2M/V).
    With M = s p q and V = p q / (2N) this reduces to Kimura's formula; with
    M taken from the full sex-specific recursion it keeps the second-order
    term (~ -p(s_f^2 + s_m^2) pq) that sex-averaging drops."""
    x = np.linspace(1e-7, 1 - 1e-7, n)
    g = 2 * M(x) / V(x)
    G = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
    lpsi = -G
    lpsi -= lpsi.max()
    psi = np.exp(lpsi)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (psi[1:] + psi[:-1]) * np.diff(x))])
    return np.interp(p0, x, cum / cum[-1])

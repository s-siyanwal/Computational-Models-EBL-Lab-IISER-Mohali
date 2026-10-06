"""Module B: mate harm as a pleiotropic side effect, not an adaptation.

A male type i has competitive mating success m_i and harm eta_i inflicted on his
current mate. Female response (Morrow, Arnqvist & Pitnick 2003; spec section 5):
    egg-laying rate  a_i = F0 * (1 - lam * eta_i)       lam >= 0
    remating hazard  r_i = r0 * (1 + rho * eta_i)        rho >= 0
Females die at rate mu. Each mating is with type i with probability
pi_i = f_i (1 + m_i) / sum_k f_k (1 + m_k). The current (last) mate sires a
share P2 of the eggs laid during his tenure; the previous mate the rest.

Closed form (exponential waiting times):
    E_i = a_i / (r_i + mu)           eggs laid while i is the current mate
    R_i = r_i / (r_i + mu)           probability she remates during i's tenure
    matings per female  M = 1 / (1 - Rbar),  first-mating share 1/M
    eggs sired per mating with a virgin    E_i + R_i (1 - P2) Ebar
    eggs sired per mating with a non-virgin   P2 E_i + R_i (1 - P2) Ebar
    male fitness  w_i ∝ (1 + m_i) * eggs sired per mating
    female fitness (lifetime eggs)  Ebar / (1 - Rbar)
The trait is autosomal and male-limited: p' = (p w_mut / wbar + p) / 2.

With rho >= 0 and lam >= 0 a harm-only mutant (m unchanged) always loses
(B1); a mutant that raises m and eta together can invade while female
fitness falls (B2). The adaptive-harm response (Johnstone & Keller 2000:
harmed females delay remating and lay faster, i.e. rho < 0, lam < 0) is
available only as an explicit counterfactual.
"""
from __future__ import annotations

import numpy as np

from . import require


def _check_signs(P, counterfactual):
    if not counterfactual and (P["rho"] < 0 or P["lam"] < 0):
        raise ValueError("rho and lam must be >= 0 (Morrow et al. 2003); negative values encode the "
                         "Johnstone-Keller adaptive-harm response and need counterfactual=True")


def fitness(freqs, m, eta, P):
    """freqs, m, eta: arrays over male types. Returns (male fitness per male,
    female lifetime eggs)."""
    f, m, eta = (np.asarray(x, float) for x in (freqs, m, eta))
    a = P["F0"] * (1 - P["lam"] * eta)
    r = P["r0"] * (1 + P["rho"] * eta)
    mu = P["mu_female"]
    E = a / (r + mu)
    R = r / (r + mu)
    pi = f * (1 + m) / np.sum(f * (1 + m))
    Ebar, Rbar = pi @ E, pi @ R
    share_first = 1 - Rbar
    p2 = P["p2_drosophila"]
    sired = share_first * (E + R * (1 - p2) * Ebar) + (1 - share_first) * (p2 * E + R * (1 - p2) * Ebar)
    return (1 + m) * sired, Ebar / (1 - Rbar)


def invade(P, flags, m_mut, eta_mut, p0=0.01, generations=300, counterfactual=False):
    """Deterministic trajectory of a mutant (m_mut, eta_mut) against the
    resident (0, 0). Returns dict(freq=array, female_eggs=array,
    s_initial=selection coefficient at p0)."""
    require(flags, "harm")
    _check_signs(P, counterfactual)
    p, traj, eggs = p0, [p0], []
    s0 = None
    for _ in range(generations):
        w, fe = fitness([1 - p, p], [0.0, m_mut], [0.0, eta_mut], P)
        wbar = (1 - p) * w[0] + p * w[1]
        if s0 is None:
            s0 = w[1] / w[0] - 1
        p = 0.5 * (p * w[1] / wbar + p)
        traj.append(p)
        eggs.append(fe)
    return dict(freq=np.array(traj), female_eggs=np.array(eggs), s_initial=float(s0))


def invasion_threshold_m(P, flags, eta):
    """Smallest gain in competitive mating success m* that lets a mutant with
    harm eta invade a harmless resident (p -> 0). B2's 'can invade' is only
    meaningful relative to this boundary: m > m*(eta)."""
    require(flags, "harm")
    _check_signs(P, False)
    w, _ = fitness([1.0, 0.0], [0.0, 0.0], [0.0, eta], P)   # w[1] = (1+0) * sired(eta) at p = 0
    return float(w[0] / w[1] - 1.0)


def invade_stochastic(rng, P, flags, m_mut, eta_mut, N=2000, p0=0.05, generations=200):
    """Wright-Fisher version (finite N) of `invade`, for drift robustness."""
    require(flags, "harm")
    _check_signs(P, False)
    p, traj = p0, [p0]
    for _ in range(generations):
        w, _ = fitness([1 - p, p], [0.0, m_mut], [0.0, eta_mut], P)
        wbar = (1 - p) * w[0] + p * w[1]
        p_exp = 0.5 * (p * w[1] / wbar + p)
        p = rng.binomial(2 * N, min(max(p_exp, 0.0), 1.0)) / (2 * N)
        traj.append(p)
    return np.array(traj)

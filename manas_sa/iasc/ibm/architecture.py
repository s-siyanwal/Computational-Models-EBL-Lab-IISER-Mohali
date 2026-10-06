"""Genetic architectures (site positions and per-copy sex-specific effects) and
the N-rescaling used to make Drosophila-scale populations laptop-sized.

Rescaling (documented in docs/03_parameters.md): for a reference population
of size N_ref, divide N by lam and multiply per-genome mutation rate U, map
length R and selection coefficients s by lam, and divide generations by lam.
This preserves N*U, N*R and N*s, i.e. the diffusion-scale parameters that
govern drift-selection-recombination dynamics (and hence Hill-Robertson
interference); the environmental noise sd is unchanged (it sets the
variance in offspring number, i.e. Ne/N).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Architecture:
    posA: np.ndarray
    posX: np.ndarray
    eff_f: np.ndarray
    eff_m: np.ndarray
    kind: np.ndarray      # 0 neutral, 1 female-limited, 2 male-limited, 3 concordant, 4 SA(f-), 5 SA(m-)
    WA: int
    WX: int


KIND_NAMES = {0: "neutral", 1: "female-limited", 2: "male-limited", 3: "sexually concordant",
              4: "SA (harms females, helps males)", 5: "SA (harms males, helps females)"}


def build(rng, n_sites_A=65536, n_sites_X=0, mix=None, gamma_shape=0.3, gamma_scale=0.05,
          sa_benefit_ratio=1.0, sc_corr=1.0):
    """mix = dict(kind -> proportion) over kinds 1..5 (and 0 neutral).
    Effects are per derived copy: deleterious part = -s, s ~ Gamma(shape, scale)
    (the 'reflected gamma' of the HRI preprint); SA beneficial part =
    +sa_benefit_ratio * s. For concordant sites the male effect is
    -s * sc_corr (sc_corr = 1: identical effects)."""
    if mix is None:
        mix = {1: 0.5, 2: 0.5}
    assert n_sites_A % 64 == 0 and n_sites_X % 64 == 0
    S = n_sites_A + n_sites_X
    kinds = np.array(sorted(mix))
    probs = np.array([mix[k] for k in kinds], float)
    probs /= probs.sum()
    kind = rng.choice(kinds, size=S, p=probs)
    s = rng.gamma(gamma_shape, gamma_scale, size=S)
    ef = np.zeros(S)
    em = np.zeros(S)
    ef[kind == 1] = -s[kind == 1]
    em[kind == 2] = -s[kind == 2]
    ef[kind == 3] = -s[kind == 3]
    em[kind == 3] = -s[kind == 3] * sc_corr
    ef[kind == 4] = -s[kind == 4]
    em[kind == 4] = +sa_benefit_ratio * s[kind == 4]
    em[kind == 5] = -s[kind == 5]
    ef[kind == 5] = +sa_benefit_ratio * s[kind == 5]
    posA = np.sort(rng.random(n_sites_A))
    posX = np.sort(rng.random(n_sites_X)) if n_sites_X else np.zeros(0)
    return Architecture(posA, posX, ef, em, kind, n_sites_A // 64, n_sites_X // 64)


@dataclass
class Rescaled:
    N: int
    n_gen: int
    U_A: float
    U_X: float
    R_A: float
    R_X: float
    gamma_scale: float
    lam: float


def rescale(N_ref, gens_ref, mu_bp, L_bp_A, R_A_ref, gamma_scale_ref, lam, L_bp_X=0, R_X_ref=0.0):
    """Return laptop-scale parameters preserving N*U, N*R, N*s."""
    return Rescaled(N=int(round(N_ref / lam)), n_gen=int(round(gens_ref / lam)),
                    U_A=mu_bp * L_bp_A * lam, U_X=mu_bp * L_bp_X * lam,
                    R_A=R_A_ref * lam, R_X=R_X_ref * lam,
                    gamma_scale=gamma_scale_ref * lam, lam=lam)

"""Deterministic cytonuclear model of 'mother's curse' (Havird et al. 2019,
Curr Biol 29:R496 -- review; we formalise the verbal argument following the
standard cytonuclear framework, e.g. Frank & Hurst 1996, Wade & Brandvain 2009).

State: zygote distribution Z[c, g]
    c in {0,1}: mitochondrial haplotype (1 = male-harming variant), maternal
    g in {0,1,2}: copies of nuclear allele R at an autosomal locus.
Life cycle: zygotes -> sex 50:50 -> sex-specific viability W[sex,c,g]
           -> random mating -> mito from mother, Mendelian nuclear alleles.

Also: an autosomal nuclear allele with the *same* sex-specific effects
(Parsons/Owen one-locus model) for the 'twofold' comparison quoted in the
review ("a nuclear-encoded variant with comparable effects on male fitness
would only be maintained by selection if it increased female fitness at least
twofold").
"""
from __future__ import annotations

import numpy as np


def fitness_tables(s_f=0.0, s_m=0.0, restore=0.0, h_R=0.5, cost_R=0.0):
    """W[sex, c, g]; sex 0 = female, 1 = male.
    females: (1 + s_f*c) * (1 - cost_R*d(g))
    males  : (1 - s_m*c*(1 - restore*d(g))) * (1 - cost_R*d(g))
    with dominance d(g) = (0, h_R, 1)."""
    d = np.array([0.0, h_R, 1.0])
    W = np.empty((2, 2, 3))
    for c in (0, 1):
        W[0, c] = (1 + s_f * c) * (1 - cost_R * d)
        W[1, c] = (1 - s_m * c * (1 - restore * d)) * (1 - cost_R * d)
    return W


def step(Z, W):
    Z = np.asarray(Z, float)
    F = Z * W[0]
    F /= F.sum()
    M = Z * W[1]
    M /= M.sum()
    # female gametes: joint (mito c, allele R)
    fR = F @ np.array([0.0, 0.5, 1.0])          # P(c, R) unnormalised by c
    fc = F.sum(1)                               # P(c)
    pm = (M.sum(0) * np.array([0.0, 0.5, 1.0])).sum()  # male R freq
    Zn = np.empty((2, 3))
    for c in range(2):
        pr = fR[c]               # P(c and R-gamete)
        pn = fc[c] - pr          # P(c and r-gamete)
        Zn[c, 0] = pn * (1 - pm)
        Zn[c, 1] = pn * pm + pr * (1 - pm)
        Zn[c, 2] = pr * pm
    return Zn / Zn.sum()


def initial(x_mito, p_R):
    """Linkage (cytonuclear) equilibrium, Hardy-Weinberg nuclear genotypes."""
    hw = np.array([(1 - p_R) ** 2, 2 * p_R * (1 - p_R), p_R ** 2])
    return np.outer([1 - x_mito, x_mito], hw)


def iterate(Z0, W, n_gen):
    Z = np.asarray(Z0, float)
    out = [Z]
    for _ in range(n_gen):
        Z = step(Z, W)
        out.append(Z)
    return np.array(out)


def mito_freq(Z):
    return np.asarray(Z)[..., 1, :].sum(-1)


def nuclear_freq(Z):
    return (np.asarray(Z).sum(-2) * np.array([0.0, 0.5, 1.0])).sum(-1)


def mito_invasion_rate(s_f):
    """Maternal transmission: x' = x(1+s_f)/(1+s_f x) => rare growth factor 1+s_f,
    independent of the male effect s_m."""
    return 1.0 + np.asarray(s_f)


def nuclear_invasion_rate(wf_het, wm_het):
    """Rare autosomal allele with heterozygote viabilities relative to the
    resident homozygote (Parsons 1961, eq. 2.4): lambda = (wf_het + wm_het)/2."""
    return 0.5 * (np.asarray(wf_het) + np.asarray(wm_het))

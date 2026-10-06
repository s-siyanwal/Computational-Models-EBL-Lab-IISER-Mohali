"""IUS selection regimes and the per-generation genotype record (genetic layer).

The genetic layer of the coupled model (docs/09_coupling.md). It evolves
populations with the unchanged Manas engine primitives (``engine._fitness``,
``_gamete``, ``_mutate``, ``_draw``) and adds three additive trait blocks at
sites that carry no fitness effect:

  clear   log clearance of P. entomophila, relative to the ancestral mean
  tol     log fecundity sensitivity to pathogen load (inverse tolerance)
  pref    male preference slope for uninfected females, relative to the
          calibration population (0 = the male cannot perceive infection)

Regimes (Chinmay 2019 p.7; Gupta 2016 Ch. 6):
  U  unhandled: no infection; the three traits are neutral. With zero trait
     effects a U generation is bit-identical to ``engine.evolve_one``.
  I  every adult infected; survival ~ Bernoulli(exp(-H)), with
     H = H1 * D**nu / clear_rel * sex_factor and the dose D reset each
     generation so that mean survival is ``target_survival`` (the protocol
     keeps mortality near 50 %); a surviving female's offspring share is
     multiplied by 1 - kappa(G, D, t_lay).
     kappa(G, D, t) = 1 - exp(-a0 * tol_rel * D**nu * exp(-cost_decay * clear_rel * t))
     (cost proportional to pathogen load, which decays from infection at a rate
     proportional to clearance; faster clearance -> lower cost at every t)

Preference sites are neutral in both regimes (every breeding female in I is a
survivor of the same infection, so a preference has no target).

This module never reads the assay layer and writes only genotype records.
Male recombination must be False (Drosophila); the runner refuses True.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import engine as E
from .architecture import Architecture

TRAITS = ("clear", "tol", "pref")
HOURS = 96.0


@dataclass
class TraitArch:
    """Per-site additive effects (per derived copy) on the three traits."""
    eff: dict                       # trait -> float array over all sites
    sites: dict                     # trait -> int array of site indices


@dataclass
class Environment:
    """Infection environment of the I regime; every number carries provenance
    in config/coupling.toml."""
    H1: float = float(np.log(2.0))  # cumulative 96 h hazard at OD 1.0 for clear_rel = 1 (50 % mortality)
    nu: float = 2.08                # dose exponent from 50 % at OD 1.0 and 80 % at OD 1.5
    a0: float = 0.0                 # fecundity sensitivity (scenario)
    cost_decay: float = 0.0         # per hour (scenario)
    t_lay: float = 105.0            # egg laying, 96 h + half of the 18 h window
    target_survival: float = 0.5
    sex_factor_f: float = 1.0       # no independent data give a genetic sex difference
    sex_factor_m: float = 1.0
    assay_dose: float = 1.5         # OD600 of the Chinmay assays
    assay_hours: float = 13.0       # 12-14 h post infection


def add_traits(rng, arch: Architecture, n_A: int, n_X: int, sd: dict) -> TraitArch:
    """Turn n_A autosomal and n_X X-linked sites per trait into trait sites:
    their fitness effects are set to 0 (kind 0) and they receive
    N(0, sd[trait]) per-copy effects on that trait."""
    S = arch.eff_f.shape[0]
    SA = arch.WA * 64
    poolA = rng.permutation(SA)
    poolX = SA + rng.permutation(arch.WX * 64)
    eff, sites = {}, {}
    for k, t in enumerate(TRAITS):
        s = np.concatenate((poolA[k * n_A:(k + 1) * n_A], poolX[k * n_X:(k + 1) * n_X])).astype(np.int64)
        e = np.zeros(S)
        e[s] = rng.normal(0.0, sd[t], s.shape[0])
        eff[t], sites[t] = e, s
    all_sites = np.concatenate(list(sites.values()))
    arch.eff_f[all_sites] = 0.0
    arch.eff_m[all_sites] = 0.0
    arch.kind[all_sites] = 0
    return TraitArch(eff, sites)


def zero_traits(n_sites: int) -> TraitArch:
    return TraitArch({t: np.zeros(n_sites) for t in TRAITS}, {t: np.zeros(0, np.int64) for t in TRAITS})


# --------------------------------------------------------------------------- numba pieces
@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _cumulate(wabs, cum, N):
    nf = N // 2
    acc = 0.0
    for i in range(nf):
        acc += wabs[i]
        cum[i] = acc
    acc = 0.0
    for i in range(nf, N):
        acc += wabs[i]
        cum[i] = acc


@njit(cache=True)
def _reproduce(G, G2, N, WA, WX, posAll, RA, RX, UA, UX, cum, male_recomb):
    """Offspring loop of engine.evolve_one, unchanged."""
    nf = N // 2
    W = WA + WX
    for j in range(N):
        mom = E._draw(cum, 0, nf)
        dad = E._draw(cum, nf, N - nf)
        mat = G2[2 * j]
        pat = G2[2 * j + 1]
        E._gamete(G[2 * mom], G[2 * mom + 1], mat, 0, WA, 0, posAll, RA, True)
        if WX > 0:
            E._gamete(G[2 * mom], G[2 * mom + 1], mat, WA, W, WA * 64, posAll, RX, True)
        E._mutate(mat, 0, WA, UA)
        if WX > 0:
            E._mutate(mat, WA, W, UX)
        E._gamete(G[2 * dad], G[2 * dad + 1], pat, 0, WA, 0, posAll, RA, male_recomb)
        E._mutate(pat, 0, WA, UA)
        if WX > 0:
            if j < nf:
                for wi in range(WA, W):
                    pat[wi] = G[2 * dad][wi]
                E._mutate(pat, WA, W, UX)
            else:
                for wi in range(WA, W):
                    pat[wi] = np.uint64(0)


# --------------------------------------------------------------------------- trait values
def trait_bv(G, N, arch, tarch, trait):
    nf = N // 2
    idx = np.arange(N)
    is_male = idx >= nf
    return E.genotype_bv(G, idx, is_male, arch.WA, arch.WX, tarch.eff[trait])


def kappa(a_rel, clear_rel, env: Environment, dose, hours):
    return 1.0 - np.exp(-env.a0 * a_rel * dose ** env.nu * np.exp(-env.cost_decay * clear_rel * hours))


def hazard96(clear_rel, is_male, env: Environment, dose):
    sexf = np.where(is_male, env.sex_factor_m, env.sex_factor_f)
    return env.H1 * dose ** env.nu / clear_rel * sexf


def _dose_for_survival(clear_rel, is_male, env):
    """Dose D (OD units) with mean survival = target (bisection on D**nu)."""
    sexf = np.where(is_male, env.sex_factor_m, env.sex_factor_f)
    k = env.H1 * sexf / clear_rel
    lo, hi = 1e-9, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if np.mean(np.exp(-k * mid)) > env.target_survival:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi)) ** (1.0 / env.nu)


# --------------------------------------------------------------------------- runner
@dataclass
class Rates:
    RA: float
    RX: float
    UA: float
    UX: float
    noise_sd: float = 1.0
    male_recombination: bool = False


@dataclass
class Ancestor:
    """Reference values of the calibration (ancestral) population."""
    mean_bv: dict = field(default_factory=lambda: {t: 0.0 for t in TRAITS})


def run_regime(G0, arch, tarch, rates: Rates, regime, n_gen, env: Environment, anc: Ancestor,
               numba_seed, rng, eff_scale=1.0, label=None, on_generation=None):
    """Evolve population G0 (2N x W uint64; females first) for n_gen
    generations under regime 'U' or 'I'. Returns the final genomes. Calls
    on_generation(summary_dict) after every generation: the per-generation
    genotype record."""
    if rates.male_recombination:
        raise ValueError("Drosophila regime: male recombination must be 0")
    if regime not in ("U", "I"):
        raise ValueError(regime)
    G = G0.copy()
    G2 = np.zeros_like(G)
    N = G.shape[0] // 2
    nf = N // 2
    is_male = np.arange(N) >= nf
    posAll = np.concatenate((arch.posA, arch.posX))
    eff_f = arch.eff_f * eff_scale
    eff_m = arch.eff_m * eff_scale
    bvf, bvm, wabs, cum = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    _seed(numba_seed)
    for gen in range(n_gen):
        E._fitness(G, N, arch.WA, arch.WX, eff_f, eff_m, rates.noise_sd, bvf, bvm, wabs)
        summ = dict(generation=gen + 1, regime=regime)
        if label:
            summ.update(label)
        need = regime == "I" or on_generation is not None
        if need:
            cl = np.exp(trait_bv(G, N, arch, tarch, "clear") - anc.mean_bv["clear"])
            tl = np.exp(trait_bv(G, N, arch, tarch, "tol") - anc.mean_bv["tol"])
            pr = np.maximum(0.0, 1.0 + trait_bv(G, N, arch, tarch, "pref") - anc.mean_bv["pref"])
        if regime == "I":
            D = _dose_for_survival(cl, is_male, env)
            surv = rng.random(N) < np.exp(-hazard96(cl, is_male, env, D))
            fec = np.where(is_male, 1.0, 1.0 - kappa(tl, cl, env, D, env.t_lay))
            wabs *= surv * fec
            if wabs[:nf].sum() <= 0 or wabs[nf:].sum() <= 0:
                raise RuntimeError("no surviving breeders of one sex")
            summ.update(dose=D, survival=float(surv.mean()))
        if on_generation is not None:
            k13 = kappa(tl[:nf], cl[:nf], env, env.assay_dose, env.assay_hours)
            summ.update(clear_rel_mean=float(np.mean(cl)), clear_rel_sd=float(np.std(cl)),
                        tol_rel_mean=float(np.mean(tl[:nf])), pref_rel_mean=float(np.mean(pr[nf:])),
                        pref_rel_sd=float(np.std(pr[nf:])), kappa_13h_mean=float(np.mean(k13)),
                        bv_f_mean=float(np.mean(bvf[:nf])), bv_m_mean=float(np.mean(bvm[nf:])))
            on_generation(summ)
        _cumulate(wabs, cum, N)
        _reproduce(G, G2, N, arch.WA, arch.WX, posAll, rates.RA, rates.RX, rates.UA, rates.UX, cum,
                   rates.male_recombination)
        G, G2 = G2, G
    return G


def sample(G, N_from, n_f, n_m, rng):
    """Individuals drawn without replacement from a population (females first,
    sexes kept), as a new population of n_f females and n_m males."""
    nf_from = N_from // 2
    fi = rng.choice(nf_from, n_f, replace=False)
    mi = nf_from + rng.choice(N_from - nf_from, n_m, replace=False)
    rows = np.concatenate([np.stack((2 * i, 2 * i + 1), 1).ravel() for i in (fi, mi)])
    return G[rows].copy()


EXPRESSED_IN = {"clear": "both", "tol": "F", "pref": "M"}


def ancestor_reference(G, arch, tarch):
    """Mean BV of each trait in the sex that expresses it (the X counts once
    in males, so a both-sex mean would bias the male- and female-limited
    traits)."""
    N = G.shape[0] // 2
    nf = N // 2
    out = {}
    for t in TRAITS:
        bv = trait_bv(G, N, arch, tarch, t)
        sel = {"both": slice(None), "F": slice(0, nf), "M": slice(nf, N)}[EXPRESSED_IN[t]]
        out[t] = float(np.mean(bv[sel]))
    return Ancestor(out)


def genotype_record(G, arch, tarch, env: Environment, anc: Ancestor, label: dict, eff_scale=1.0):
    """Per-individual genotype record (docs/09 9.1). tarch=None means the
    population carries no assay traits (e.g. a saved Manas population); the
    trait fields are then NaN and the assay layer keeps its constants."""
    import pandas as pd
    N = G.shape[0] // 2
    nf = N // 2
    idx = np.arange(N)
    is_male = idx >= nf
    bf = E.genotype_bv(G, idx, is_male, arch.WA, arch.WX, arch.eff_f * eff_scale)
    bm = E.genotype_bv(G, idx, is_male, arch.WA, arch.WX, arch.eff_m * eff_scale)
    sa = np.isin(arch.kind, (4, 5))
    saf = E.genotype_bv(G, idx, is_male, arch.WA, arch.WX, np.where(sa, arch.eff_f, 0.0) * eff_scale)
    sam = E.genotype_bv(G, idx, is_male, arch.WA, arch.WX, np.where(sa, arch.eff_m, 0.0) * eff_scale)
    rec = dict(id=idx, sex=np.where(is_male, "M", "F"), w_f=np.exp(bf), w_m=np.exp(bm),
               w_adult=np.where(is_male, np.exp(bm), np.exp(bf)), sa_load_f=saf, sa_load_m=sam)
    if tarch is None:
        for k in ("clear_rel", "tol_rel", "beta_rel", "kappa_13h", "kappa_lay", "H96_assay"):
            rec[k] = np.full(N, np.nan)
    else:
        cl = np.exp(trait_bv(G, N, arch, tarch, "clear") - anc.mean_bv["clear"])
        tl = np.exp(trait_bv(G, N, arch, tarch, "tol") - anc.mean_bv["tol"])
        pr = np.maximum(0.0, 1.0 + trait_bv(G, N, arch, tarch, "pref") - anc.mean_bv["pref"])
        rec.update(clear_rel=cl, tol_rel=np.where(is_male, np.nan, tl), beta_rel=np.where(is_male, pr, np.nan),
                   kappa_13h=np.where(is_male, np.nan, kappa(tl, cl, env, env.assay_dose, env.assay_hours)),
                   kappa_lay=np.where(is_male, np.nan, kappa(tl, cl, env, env.assay_dose, env.t_lay)),
                   H96_assay=hazard96(cl, is_male, env, env.assay_dose))
    df = pd.DataFrame(rec)
    for k, v in label.items():
        df.insert(0, k, v)
    return df

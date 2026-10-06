"""Coupler: genetic-layer genotype record -> assay inputs (docs/09_coupling.md).

One direction only. The coupler reads a record and never writes into a record
or a population. Flag ``coupling`` (default off): with it off the assay layer
uses the regime's global constants, exactly as before.

Mapping, per vial of a two-choice assay (one infected and one sham female of
the female population, one male of the male population):
  infected female  q_inf = 1 - kappa_13h(G)      (cue on; 1 if cue off)
  sham female      q_sham = 1                    (not infected: kappa = 0)
  male             b = beta0 * depletion * beta_rel(G)
                   (beta0 from the Byrne & Rice calibration, never retuned;
                   beta_rel = 0 is a male who cannot perceive infection)
A record field that is NaN (the population carries no assay traits) falls
back to the regime constant (kappa, beta_rel = 1).

The default cue is q = 1 - kappa, so an allele that changes only adult
fitness (e.g. a sexually antagonistic allele) cannot change courtship.
``cue_includes_baseline_fecundity`` (off) is the third legal channel: q is
then scaled by the female's own w_f relative to the population mean.

Nothing here feeds telegony.py: the stepfather's genotype, SA alleles
included, never reaches offspring survival.
"""
from __future__ import annotations

import numpy as np

from . import require
from . import choice as C
from .quality import Female, perceived_quality, receptivity


def load_saved_population(path, rep=0, label=None):
    """Genotype record of a saved Manas population (results/pops/*.npz).
    These populations carry no assay traits, so every trait field is NaN."""
    from iasc.ibm.architecture import Architecture
    from iasc.ibm.regime import Ancestor, Environment, genotype_record
    z = np.load(path)
    arch = Architecture(z["posA"], z["posX"], z["eff_f"], z["eff_m"], z["kind"], int(z["WA"]), int(z["WX"]))
    return genotype_record(z["genomes"][rep], arch, None, Environment(), Ancestor(), label or {"population": "saved"})


def _pick(record, rng, n, sex, **where):
    sub = record[record.sex == sex]
    for k, v in where.items():
        sub = sub[sub[k] == v]
    if len(sub) == 0:
        raise ValueError(f"no {sex} genotypes in record for {where}")
    return sub.iloc[rng.integers(0, len(sub), n)]


def _q(kappa, P, w_rel=None):
    if not P.get("cue_on", True):
        return np.ones_like(kappa)
    q = 1.0 - kappa
    if P.get("cue_includes_baseline_fecundity", False) and w_rel is not None:
        q = q * w_rel
    return q


def vial_inputs(record, P, flags, rng_sample, n, female_where, male_where):
    """Arrays (q_sham, q_inf, b) for n vials."""
    require(flags, "coupling")
    fem = _pick(record, rng_sample, n, "F", **female_where)
    sham = _pick(record, rng_sample, n, "F", **female_where)
    mal = _pick(record, rng_sample, n, "M", **male_where)
    k = fem.kappa_13h.to_numpy(float)
    k = np.where(np.isfinite(k), k, P["kappa"])
    br = mal.beta_rel.to_numpy(float)
    br = np.where(np.isfinite(br), br, 1.0)
    w_mean = record[record.sex == "F"].w_f.mean()
    q_inf = _q(k, P, fem.w_f.to_numpy(float) / w_mean)
    q_sham = _q(np.zeros(n), P, sham.w_f.to_numpy(float) / w_mean)
    if P.get("couple_quality_to_survival", False):
        raise NotImplementedError("survival coupling of quality is off in every audited regime")
    b = C.beta(P, P["male_depleted"]) * br
    return q_sham, q_inf, b


def simulate_chinmay_study(rng, P, flags, record, rng_sample, population_key="population"):
    """Thesis Ch. 3 design (Table 3.2 sizes) with female and male genotypes
    drawn from the record: block b uses record rows with block == b and
    population == 'I' or 'U'. The assay RNG stream ``rng`` is used exactly as
    in choice.simulate_chinmay_study; genotypes are drawn with rng_sample."""
    require(flags, "choice")
    require(flags, "quality")
    require(flags, "coupling")
    parts = {k: [] for k in ("block", "trt", "female", "male", "courted", "cf_sham", "lat_sham", "lat_inf", "cm_sham")}
    has_block = "block" in record.columns
    for block, trt, _, n in C.TABLE_3_2:
        fem_gt, male_gt = trt[0], trt[-1]
        fw = {population_key: fem_gt}
        mw = {population_key: male_gt}
        if has_block:
            fw["block"] = block
            mw["block"] = block
        q_s, q_i, b = vial_inputs(record, P, flags, rng_sample, n, fw, mw)
        mult = np.exp(P["male_genotype_latency_effect"] * (1 if male_gt == "I" else -1)
                      + P["female_genotype_latency_effect"] * (1 if fem_gt == "I" else -1))
        r = C.simulate_two_choice(rng, n, q_s, q_i, b, P, lat_mult=mult)
        parts["block"].append(np.full(n, block))
        parts["trt"].append(np.full(n, trt))
        parts["female"].append(np.full(n, fem_gt))
        parts["male"].append(np.full(n, male_gt))
        for k in ("courted", "cf_sham", "lat_sham", "lat_inf", "cm_sham"):
            parts[k].append(r[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def group_mating_vial(rng, q_a, q_b, b_males, n_each, minutes, acceptance, latency_mean):
    """choice.group_mating_vial with one beta per male. With all betas equal
    it calls the original function (identical RNG use)."""
    b_males = np.asarray(b_males, float)
    if np.all(b_males == b_males[0]):
        return C.group_mating_vial(rng, q_a, q_b, float(b_males[0]), b_males.size, n_each, minutes, acceptance,
                                   latency_mean)
    avail = np.array([n_each, n_each])
    mated = np.zeros(2, int)
    free = list(range(b_males.size))
    p_try = 1 - np.exp(-1.0 / latency_mean)
    for _ in range(int(minutes)):
        if not free or avail.sum() == 0:
            break
        for _ in range(rng.binomial(len(free), p_try)):
            if avail.sum() == 0 or not free:
                break
            m = free[rng.integers(len(free))]
            w = avail * np.exp(b_males[m] * np.array([q_a, q_b]))
            k = 0 if rng.random() < w[0] / w.sum() else 1
            if rng.random() < acceptance:
                avail[k] -= 1
                mated[k] += 1
                free.remove(m)
    return int(mated[0]), int(mated[1])


def simulate_khan(rng, P, flags, record, rng_sample, n_experiments=4):
    """Khan & Prasad (2013) design with male betas from the record (kappa is
    Khan's measured regime value; females are not drawn from the record)."""
    require(flags, "choice")
    require(flags, "quality")
    require(flags, "coupling")
    q_s = perceived_quality(Female("LH", False, P["decapitated"]), P)
    q_i = perceived_quality(Female("LH", True, P["decapitated"]), P)
    acc = receptivity(Female("LH", False, P["decapitated"]), P)
    b0 = C.beta(P, P["male_depleted"])
    scores, vial_scores = [], []
    for _ in range(n_experiments):
        s = []
        for _ in range(int(P["vials_per_experiment"])):
            br = _pick(record, rng_sample, int(P["group_males"]), "M").beta_rel.to_numpy(float)
            br = np.where(np.isfinite(br), br, 1.0)
            ms, mi = group_mating_vial(rng, q_s, q_i, b0 * br, P["group_females_each"], P["group_minutes"], acc,
                                       P["latency_mean_min"])
            if mi + ms:
                s.append(mi / (mi + ms))
        scores.append(float(np.mean(s)))
        vial_scores.append(s)
    return dict(bias_scores=scores, vial_scores=vial_scores)      # vial_scores: one list per experiment

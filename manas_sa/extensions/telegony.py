"""Module C: telegony switch (first-male / stepfather effects).

A female mates first with male 1 (the stepfather) and then with male 2.
A non-genetic semen state written by male 1 can reach the ova only if the
female tract is NOT compartmentalised (immature, permeable ovules: the
Telostylinus template of Crean, Kopps & Bonduriansky 2014). For Drosophila the
compartmentalisation flag defaults on (Chinmay 2019 p.32, an author
hypothesis), so male 1 cannot shift offspring phenotype; only the genetic sire
and the offspring's sex matter.

Mother's curse (Havird et al. 2019) is a genetic cytonuclear effect and is NOT
used here: this module has no mitochondrial state and imports nothing from the
cytonuclear models.

When the module is off no non-genetic state is allocated (C4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import require
from . import stats as S

# census times read from the x-axis of thesis Figs 4.1-4.2 (p.30, PDF p.48), hours
CENSUS_H = np.array([6, 12, 15, 18, 20, 22, 23, 24.5, 26.5, 29.5, 33.5, 37.5, 43.5, 49.5, 54, 60,
                     65.5, 71.5, 77.5, 83.5, 91.5, 97.5])
STEPFATHERS = ("I", "U", "S")
TIMEPOINTS = ("6h", "10h")


@dataclass
class MatedFemale:
    """A female after her matings. The ``semen_state`` attribute (the condition
    signal written by male 1) is attached only when the telegony module is on;
    it is not a declared field, so with the module off it does not exist."""
    first_male: str
    second_male: str


def mate_twice(first_male, second_male, flags, first_male_condition=0.0):
    """Returns a MatedFemale; allocates the non-genetic state only if the
    telegony module is on (C4)."""
    f = MatedFemale(first_male, second_male)
    if flags.get("telegony", False):
        f.semen_state = {"first_male_condition": float(first_male_condition)}
    return f


def first_male_shift(female, P, effect_size):
    """Offspring phenotype shift caused by male 1's semen. Zero whenever the
    tract is compartmentalised or no state exists."""
    state = getattr(female, "semen_state", None)
    if state is None or P["compartmentalization"]:
        return 0.0
    return effect_size * state["first_male_condition"]


def _hazard(mort96):
    return -np.log(1 - mort96) / 96.0


def simulate_chinmay_telegony(rng, P, flags, stepfather_condition=None):
    """One block of thesis Ch. 4: LHst females mated first to I, U or S males
    (6 h or 10 h after heat-killed P. entomophila), then to LHst males;
    scarlet-eyed (second-male) offspring infected, censused to 97.5 h.
    stepfather_condition: {I: c, U: c, S: c} condition signal per stepfather
    population (only matters if compartmentalization is off). Returns arrays
    (time, event, stepfather, timepoint, sex) for infected offspring."""
    require(flags, "telegony")
    cond = stepfather_condition or {k: 0.0 for k in STEPFATHERS}
    h_f = _hazard(P["mortality_96h_daughters"])
    rows = []
    n = int(P["n_infected_per_cell"])
    for sf in STEPFATHERS:
        for tp in TIMEPOINTS:
            fem = mate_twice(sf, "LHst", flags, cond[sf])
            shift = first_male_shift(fem, P, P["semen_effect_sd"])   # log-hazard shift
            for sex in ("F", "M"):
                h = h_f * (P["sex_hazard_ratio_male"] if sex == "M" else 1.0) * np.exp(-shift)
                t_death = rng.exponential(1 / h, n)
                idx = np.searchsorted(CENSUS_H, t_death)              # first census after death
                event = idx < len(CENSUS_H)
                t_obs = np.where(event, CENSUS_H[np.minimum(idx, len(CENSUS_H) - 1)], CENSUS_H[-1])
                for i in range(n):
                    rows.append((t_obs[i], event[i], sf, tp, sex))
    t, e, sf, tp, sx = map(np.array, zip(*rows))
    return dict(time=t.astype(float), event=e.astype(bool), stepfather=sf, timepoint=tp, sex=sx)


def analyse_telegony(d):
    """Thesis Table 4.2 (Wald tests from a Cox model with population, time point,
    gender and interactions) and Table 4.1 (log-rank per sex x time point)."""
    out = {}
    w = S.cox_wald(d["time"], d["event"], {"Population": d["stepfather"], "TimePoint": d["timepoint"],
                                           "Gender": d["sex"]})
    for term, (chi2, df, p) in w.items():
        out[("wald", term)] = (chi2, df, p)
    for sex in ("F", "M"):
        for tp in TIMEPOINTS:
            m = (d["sex"] == sex) & (d["timepoint"] == tp)
            out[("logrank", sex, tp)] = S.logrank(d["time"][m], d["event"][m], d["stepfather"][m])
    return out


def simulate_telostylinus(rng, P, flags, n_families=48, offspring=10, family_share=0.70):
    """Crean et al. (2014) design: first and second males of high or low
    condition (fully crossed), 48 genotyped families. Second male sires a share
    p2 of offspring. Offspring size in units of the total SD = first-male semen
    shift + family effect (variance share 0.70, Crean Table 1) + residual;
    acquired condition is not genetic, so the genetic sire adds nothing.
    Returns dict(p2_observed, effect_first, p_first, effect_second, p_second)."""
    family_sd, resid_sd = np.sqrt(family_share), np.sqrt(1 - family_share)
    require(flags, "telegony")
    from scipy import stats as st
    fam_rows = []
    for k in range(n_families):
        c1, c2 = k % 2, (k // 2) % 2
        fem = mate_twice("high" if c1 else "low", "high" if c2 else "low", flags, c1)
        shift = first_male_shift(fem, P, P["semen_effect_sd"])
        sire2 = rng.random(offspring) < P["p2_second_male"]
        size = shift + rng.normal(0, family_sd) + rng.normal(0, resid_sd, offspring)
        fam_rows.append((c1, c2, sire2.mean(), size[sire2].mean() if sire2.any() else np.nan))
    c1, c2, p2, sz = map(np.array, zip(*fam_rows))
    ok = np.isfinite(sz)
    X = np.column_stack([np.ones(ok.sum()), c1[ok], c2[ok]])
    beta, res, *_ = np.linalg.lstsq(X, sz[ok], rcond=None)
    resid = sz[ok] - X @ beta
    dof = ok.sum() - 3
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    t1, t2 = beta[1] / np.sqrt(cov[1, 1]), beta[2] / np.sqrt(cov[2, 2])
    return dict(p2_observed=float(p2.mean()), effect_first=float(beta[1]), p_first=float(2 * st.t.sf(abs(t1), dof)),
                effect_second=float(beta[2]), p_second=float(2 * st.t.sf(abs(t2), dof)))

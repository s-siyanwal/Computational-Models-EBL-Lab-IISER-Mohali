"""The estimators of Geeta Arun et al. (2022), implemented once and applied to
both the authors' raw data and to simulated assays (same columns:
Sex, SexRatio, Day, Line, Fitness)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SR = ("M", "E", "F")


# Flag F12: the Methods say male fitness was arcsine-square-root transformed,
# but every Table 2A line-mean statistic (r_w,g,mf, SA proportion, all
# across-sex-ratio correlations) is reproduced to 4 decimals only WITHOUT the
# transform; with it, r = 0.382/0.419/0.236. We therefore score exactly as the
# published numbers were computed.
ARCSINE_MALES = False


def line_means(d, transform=ARCSINE_MALES):
    """(male) arcsine-sqrt -> divide by day mean -> line mean per day -> mean of day means."""
    d = d.copy()
    if transform:
        m = d.Sex == "Male"
        d.loc[m, "Fitness"] = np.arcsin(np.sqrt(np.clip(d.loc[m, "Fitness"], 0, 1)))
    d["w"] = d.Fitness / d.groupby(["Sex", "SexRatio", "Day"]).Fitness.transform("mean")
    per_day = d.groupby(["Sex", "SexRatio", "Day", "Line"]).w.mean().reset_index()
    return per_day.groupby(["Sex", "SexRatio", "Line"]).w.mean().reset_index()


def paper_stats(lm):
    """r_w,g,mf, SA proportion (45 deg rotation) and across-sex-ratio
    correlations from scaled-and-centred line means."""
    out = {}
    wide = lm.pivot_table(index="Line", columns=["Sex", "SexRatio"], values="w")
    z = (wide - wide.mean()) / wide.std(ddof=1)
    for s in SR:
        if ("Female", s) not in z or ("Male", s) not in z:
            continue
        f, m = z[("Female", s)], z[("Male", s)]
        ok = f.notna() & m.notna()
        out[("r_wgmf", s)] = float(np.corrcoef(f[ok], m[ok])[0, 1])
        wc = (f[ok] + m[ok]) / np.sqrt(2)
        wa = (-f[ok] + m[ok]) / np.sqrt(2)
        out[("prop_SA", s)] = float(wa.var() / (wa.var() + wc.var()))
    for sex, key in (("Female", "r_female_across"), ("Male", "r_male_across")):
        for a, b in (("M", "F"), ("M", "E"), ("F", "E")):
            if (sex, a) in z and (sex, b) in z:
                x, y = z[(sex, a)], z[(sex, b)]
                ok = x.notna() & y.notna()
                out[(key, f"{a}-{b}")] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return out


def relative_stats(raw):
    """Thesis Ch. 4 line variances of relative fitness and the within-/cross-sex
    (co)variances that enter the sex-specific Fundamental Theorem
    (Robertson-style decomposition of the response of fitness itself)."""
    per_day = raw.groupby(["Sex", "SexRatio", "Day", "Line"]).Fitness.mean().reset_index()
    lm = per_day.groupby(["Sex", "SexRatio", "Line"]).Fitness.mean().reset_index()
    lm["rel"] = lm.Fitness / lm.groupby(["Sex", "SexRatio"]).Fitness.transform("mean")
    wide = lm.pivot_table(index="Line", columns=["Sex", "SexRatio"], values="rel")
    out = {}
    for s in SR:
        f, m = wide[("Female", s)], wide[("Male", s)]
        ok = f.notna() & m.notna()
        vf, vm = f.var(ddof=1), m.var(ddof=1)
        c = np.cov(f[ok], m[ok])[0, 1]
        out[("linevar_rel_female", s)] = float(vf)
        out[("linevar_rel_male", s)] = float(vm)
        out[("linecov_rel_mf", s)] = float(c)
        out[("cross_over_within_female", s)] = float(c / vf)
        out[("cross_over_within_male", s)] = float(c / vm)
    return out


def within_cell_cv2(raw):
    """Mean squared CV of vial fitness within line x day cells (noise target)."""
    out = {}
    for (sex, s), d in raw.groupby(["Sex", "SexRatio"]):
        g = d.groupby(["Day", "Line"]).Fitness
        cv2 = (g.var(ddof=1) / g.mean() ** 2).replace([np.inf], np.nan)
        out[(f"within_cv2_{sex.lower()}", s)] = float(np.nanmean(cv2))
    return out


def all_stats(raw):
    st = paper_stats(line_means(raw))
    st.update(relative_stats(raw))
    st.update(within_cell_cv2(raw))
    return st

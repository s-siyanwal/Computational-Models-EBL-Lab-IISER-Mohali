"""Recompute the published statistics of Geeta Arun et al. (2022, BMC Ecol Evol 22:38)
from their own raw data (Additional file 2 = 12862_2022_1992_MOESM3_ESM.xlsx),
following the Methods section ("Statistical analysis") as literally as possible.

Outputs results/bmc2022_recomputed.csv: estimate + bootstrap 95% CI for
  * intersexual genetic correlation (line averages)       Table 2A
  * proportion of sexually antagonistic fitness variation  Table 2A
  * across-sex-ratio correlations for each sex             Table 2A
  * line variance of relative fitness (thesis Ch. 4, Fig 4.1)
  * within-sex and cross-sex covariances of relative fitness (FTNS/Robertson
    decomposition of the expected response of fitness itself)
Reported values are attached for side-by-side comparison (claim type A vs B).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), "experimental_papers_manas", "12862_2022_1992_MOESM3_ESM.xlsx")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

SR = ["M", "E", "F"]                    # male biased, equal, female biased
SR_NAME = {"M": "male biased", "E": "equal", "F": "female biased"}

# Reported values (Table 2A of the paper; thesis Ch. 4 for line variances)
REPORTED = {
    ("r_wgmf", "M"): (0.3805, 0.2992, 0.5283), ("r_wgmf", "E"): (0.4027, 0.3140, 0.5526),
    ("r_wgmf", "F"): (0.2515, 0.1198, 0.4502),
    ("prop_SA", "M"): (0.3097, 0.2358, 0.3504), ("prop_SA", "E"): (0.2986, 0.2237, 0.3430),
    ("prop_SA", "F"): (0.3742, 0.2749, 0.4401),
    ("r_female_across", "M-F"): (0.7688, 0.7442, 0.8497), ("r_female_across", "M-E"): (0.7493, 0.7213, 0.8368),
    ("r_female_across", "F-E"): (0.8421, 0.8403, 0.8956),
    ("r_male_across", "M-F"): (0.5567, 0.4997, 0.7262), ("r_male_across", "M-E"): (0.6995, 0.6755, 0.8018),
    ("r_male_across", "F-E"): (0.5415, 0.4664, 0.7417),
    ("linevar_rel_male", "M"): (0.2926, 0.1954, 0.4859), ("linevar_rel_male", "E"): (0.1591, 0.1062, 0.2642),
    ("linevar_rel_male", "F"): (0.1010, 0.0674, 0.1678),
    ("linevar_rel_female", "M"): (0.0532, 0.0355, 0.0884), ("linevar_rel_female", "E"): (0.0321, 0.0214, 0.0533),
    ("linevar_rel_female", "F"): (0.0254, 0.0170, 0.0422),
}


def load():
    raw = pd.read_excel(DATA, header=1)
    raw.columns = ["Sex", "SexRatio", "Day", "Line", "Fitness"]
    raw = raw.dropna(subset=["Fitness"])
    raw["Fitness"] = raw.Fitness.astype(float)
    raw["Line"] = raw.Line.astype(int)
    return raw


# Flag F12: the Methods say male fitness was arcsine-square-root transformed,
# but every Table 2A line-mean statistic (r_w,g,mf, SA proportion, all
# across-sex-ratio correlations) is reproduced to 4 decimals only WITHOUT the
# transform; with it, r = 0.382/0.419/0.236. We therefore score exactly as the
# published numbers were computed.
ARCSINE_MALES = False


def line_means(d, transform=ARCSINE_MALES):
    """Paper pipeline: (male) arcsin-sqrt -> divide by day mean -> line mean per day
    -> mean of the day means.  Returns DataFrame[Sex, SexRatio, Line, w]."""
    d = d.copy()
    if transform:
        m = d.Sex == "Male"
        d.loc[m, "Fitness"] = np.arcsin(np.sqrt(d.loc[m, "Fitness"].clip(0, 1)))
    d["w"] = d.Fitness / d.groupby(["Sex", "SexRatio", "Day"]).Fitness.transform("mean")
    per_day = d.groupby(["Sex", "SexRatio", "Day", "Line"]).w.mean().reset_index()
    return per_day.groupby(["Sex", "SexRatio", "Line"]).w.mean().reset_index()


def stats_from_linemeans(lm):
    out = {}
    wide = lm.pivot_table(index="Line", columns=["Sex", "SexRatio"], values="w")
    z = (wide - wide.mean()) / wide.std(ddof=1)            # scaled & centred per sex x sex ratio
    for s in SR:
        f, m = z[("Female", s)], z[("Male", s)]
        ok = f.notna() & m.notna()
        out[("r_wgmf", s)] = np.corrcoef(f[ok], m[ok])[0, 1]
        wc = (f[ok] + m[ok]) / np.sqrt(2)
        wa = (-f[ok] + m[ok]) / np.sqrt(2)
        out[("prop_SA", s)] = wa.var() / (wa.var() + wc.var())
    for sex, key in (("Female", "r_female_across"), ("Male", "r_male_across")):
        for a, b in (("M", "F"), ("M", "E"), ("F", "E")):
            x, y = z[(sex, a)], z[(sex, b)]
            ok = x.notna() & y.notna()
            out[(key, f"{a}-{b}")] = np.corrcoef(x[ok], y[ok])[0, 1]
    return out


def relative_stats(raw):
    """Thesis Ch. 4: relative fitness = line average / mean (sex x sex ratio);
    line variance per sex x sex ratio. Also cross-sex covariance of relative
    fitness (Robertson / sex-specific FTNS decomposition)."""
    d = raw.copy()
    per_day = d.groupby(["Sex", "SexRatio", "Day", "Line"]).Fitness.mean().reset_index()
    lm = per_day.groupby(["Sex", "SexRatio", "Line"]).Fitness.mean().reset_index()
    lm["rel"] = lm.Fitness / lm.groupby(["Sex", "SexRatio"]).Fitness.transform("mean")
    wide = lm.pivot_table(index="Line", columns=["Sex", "SexRatio"], values="rel")
    out = {}
    for s in SR:
        f, m = wide[("Female", s)], wide[("Male", s)]
        ok = f.notna() & m.notna()
        out[("linevar_rel_female", s)] = f.var(ddof=1)
        out[("linevar_rel_male", s)] = m.var(ddof=1)
        out[("linecov_rel_mf", s)] = np.cov(f[ok], m[ok])[0, 1]
        # sex-specific FTNS (eq. 2 of Geeta Arun 2025 bioRxiv) on line-level
        # (co)variances: within-sex and cross-sex parts of the expected response
        out[("within_sex_response_female", s)] = 0.5 * f.var(ddof=1)
        out[("cross_sex_response_female", s)] = 0.5 * np.cov(f[ok], m[ok])[0, 1]
        out[("within_sex_response_male", s)] = 0.5 * m.var(ddof=1)
        out[("cross_sex_response_male", s)] = 0.5 * np.cov(f[ok], m[ok])[0, 1]
        out[("cross_over_within_female", s)] = np.cov(f[ok], m[ok])[0, 1] / f.var(ddof=1)
        out[("cross_over_within_male", s)] = np.cov(f[ok], m[ok])[0, 1] / m.var(ddof=1)
    return out


def bootstrap(raw, B=10000, seed=20220315, identity=False):
    """Stratified bootstrap: resample vials with replacement within each
    sex x line x day (x sex ratio) cell, as described in Methods.
    Vectorised (np.bincount) re-implementation of the pandas pipeline above;
    test: identity=True reproduces the point estimates."""
    rng = np.random.default_rng(seed)
    d = raw.sort_values(["Sex", "SexRatio", "Day", "Line"]).reset_index(drop=True)
    cell = d.groupby(["Sex", "SexRatio", "Day", "Line"], sort=False).ngroup().values
    ncell = cell.max() + 1
    start = np.searchsorted(cell, np.arange(ncell))
    size = np.bincount(cell, minlength=ncell)
    ck = d.groupby(["Sex", "SexRatio", "Day", "Line"], sort=False).size().reset_index()
    dayg = ck.groupby(["Sex", "SexRatio", "Day"], sort=False).ngroup().values       # cell -> day group
    ck["Line"] = ck.Line.astype(int)
    lines = np.sort(ck.Line.unique())
    li = np.searchsorted(lines, ck.Line.values)
    combos = [(sx, s) for sx in ("Female", "Male") for s in SR]
    combo_of = {c: k for k, c in enumerate(combos)}
    co = np.array([combo_of[(a, b)] for a, b in zip(ck.Sex, ck.SexRatio)])
    nL, nC = len(lines), len(combos)
    F = d.Fitness.values.astype(float)
    T = np.where(d.Sex.values == "Male", np.arcsin(np.sqrt(np.clip(F, 0, 1))), F) if ARCSINE_MALES else F
    rec = []
    for b in range(B):
        src = np.arange(len(cell)) if identity else start[cell] + (rng.random(len(cell)) * size[cell]).astype(int)
        f, t = F[src], T[src]
        cm_t = np.bincount(cell, t, ncell) / size                                   # cell mean (transformed)
        cm_f = np.bincount(cell, f, ncell) / size                                   # cell mean (raw)
        # day mean over vials of the day group (sum over cells / n vials)
        ng = dayg.max() + 1
        day_mean = np.bincount(dayg, cm_t * size, ng) / np.bincount(dayg, size, ng)
        w_cell = cm_t / day_mean[dayg]
        key = co * nL + li
        cnt = np.bincount(key, minlength=nC * nL)
        W = (np.bincount(key, w_cell, nC * nL) / np.where(cnt, cnt, np.nan)).reshape(nC, nL)
        R = (np.bincount(key, cm_f, nC * nL) / np.where(cnt, cnt, np.nan)).reshape(nC, nL)
        R = R / np.nanmean(R, 1, keepdims=True)
        st = {}
        Z = (W - np.nanmean(W, 1, keepdims=True)) / np.nanstd(W, 1, ddof=1, keepdims=True)

        def corr(a, c):
            ok = ~np.isnan(a) & ~np.isnan(c)
            return np.corrcoef(a[ok], c[ok])[0, 1]
        for s in SR:
            zf, zm = Z[combo_of[("Female", s)]], Z[combo_of[("Male", s)]]
            ok = ~np.isnan(zf) & ~np.isnan(zm)
            st[("r_wgmf", s)] = corr(zf, zm)
            wc, wa = (zf[ok] + zm[ok]) / np.sqrt(2), (-zf[ok] + zm[ok]) / np.sqrt(2)
            st[("prop_SA", s)] = wa.var(ddof=1) / (wa.var(ddof=1) + wc.var(ddof=1))
            rf, rm = R[combo_of[("Female", s)]], R[combo_of[("Male", s)]]
            ok2 = ~np.isnan(rf) & ~np.isnan(rm)
            vf, vm = np.nanvar(rf, ddof=1), np.nanvar(rm, ddof=1)
            c = np.cov(rf[ok2], rm[ok2])[0, 1]
            st.update({("linevar_rel_female", s): vf, ("linevar_rel_male", s): vm, ("linecov_rel_mf", s): c,
                       ("within_sex_response_female", s): vf / 2, ("cross_sex_response_female", s): c / 2,
                       ("within_sex_response_male", s): vm / 2, ("cross_sex_response_male", s): c / 2,
                       ("cross_over_within_female", s): c / vf, ("cross_over_within_male", s): c / vm})
        for sex, k in (("Female", "r_female_across"), ("Male", "r_male_across")):
            for a, c in (("M", "F"), ("M", "E"), ("F", "E")):
                st[(k, f"{a}-{c}")] = corr(Z[combo_of[(sex, a)]], Z[combo_of[(sex, c)]])
        rec.append(st)
        if b % 2000 == 0:
            print(f"  bootstrap {b}/{B}", flush=True)
    return pd.DataFrame(rec)


def main(B=10000):
    raw = load()
    print(f"rows: {len(raw)}; lines: {raw.Line.nunique()}; days: {sorted(raw.Day.unique())}")
    point = stats_from_linemeans(line_means(raw))
    point.update(relative_stats(raw))
    boot = bootstrap(raw, B=B)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        rep = REPORTED.get(k)
        # "basic" (reflected) interval 2*theta - q: the vial-resampling bootstrap
        # attenuates correlations, so percentile intervals sit below the estimate
        rows.append(dict(statistic=k[0], level=k[1], recomputed=v, boot_lo=lo, boot_hi=hi,
                         basic_lo=2 * v - hi, basic_hi=2 * v - lo,
                         reported=rep[0] if rep else np.nan, reported_lo=rep[1] if rep else np.nan,
                         reported_hi=rep[2] if rep else np.nan))
    # differences M - F (paper reports CI of difference)
    for stat in ("r_wgmf", "prop_SA"):
        diff = boot[(stat, "M")] - boot[(stat, "F")]
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        rep = {"r_wgmf": (-0.0721, 0.3507), "prop_SA": (-0.1753, 0.0360)}[stat]
        rows.append(dict(statistic=stat + "_diff", level="M-F", recomputed=point[(stat, "M")] - point[(stat, "F")],
                         boot_lo=lo, boot_hi=hi, reported=np.nan, reported_lo=rep[0], reported_hi=rep[1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "bmc2022_recomputed.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main(B=int(sys.argv[1]) if len(sys.argv) > 1 else 10000)

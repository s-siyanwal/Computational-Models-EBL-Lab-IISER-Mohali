"""Formal tests of whether the model predicts / explains each scorable result.
Writes outputs/hypothesis_tests.csv.

For every result: does the observed value look like a draw from the model's own
sampling distribution (predictive p-value, two-sided unless stated)? Where a
rival hypothesis or a trivial baseline exists, which does the data favour
(likelihood ratio; error relative to the baseline)? Benjamini-Hochberg FDR
over all predictive p-values. Each row carries the evidence type from the
audit (docs/07): a 'consistent' verdict on a by-construction row is not
evidence.

usage: python analysis/hypothesis_tests.py [--reps 300]
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from extensions import choice as C  # noqa: E402
from extensions import stats as S  # noqa: E402
from extensions import telegony as T  # noqa: E402
from extensions.config import load_regime, values  # noqa: E402
from iasc import params as IP  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
RAW = os.path.join(os.path.dirname(ROOT), "experimental_papers_manas", "12862_2022_1992_MOESM3_ESM.xlsx")
SEED = 31415


def pred_p(sim, obs, side="two"):
    """Predictive (Monte Carlo) p-value of obs under simulated values."""
    sim = np.asarray(sim, float)
    lo = (np.sum(sim <= obs) + 1) / (len(sim) + 1)
    hi = (np.sum(sim >= obs) + 1) / (len(sim) + 1)
    return {"two": min(1.0, 2 * min(lo, hi)), "greater": hi, "less": lo}[side]


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    IP.below_normal_priority()
    R = args.reps
    ius, khan = load_regime("ius"), load_regime("khan")
    P = values(ius)
    Pk = dict(P)
    Pk.update(values(khan))
    F = ius["modules"]
    rows = []

    def add(**kw):
        rows.append(kw)

    # ------------------------------------------------------------ Chinmay courtship
    sims0 = [C.analyse_chinmay(C.simulate_chinmay_study(np.random.default_rng(SEED + r), P, F)) for r in range(R)]
    sims1 = [C.analyse_chinmay(C.simulate_chinmay_study(np.random.default_rng(SEED + 5000 + r), P, F, kappa=Pk["kappa"]))
             for r in range(R)]
    K = sum(k for _, _, k, _ in C.TABLE_3_2)
    N = sum(n for *_, n in C.TABLE_3_2)
    p1 = float(C.p_first(1.0, 1 - Pk["kappa"], C.beta(P, True)))
    llr = st.binom.logpmf(K, N, 0.5) - st.binom.logpmf(K, N, p1)
    add(id="H1", target="Chinmay courts-first (pooled 240/463)", hypothesis="kappa = 0 (no fecundity cost) vs kappa = 0.4 (Khan-sized), calibrated beta",
        test="binomial likelihood ratio", statistic=f"p(kappa=0) = 0.500, p(kappa=0.4) = {p1:.3f}; log LR = {llr:.2f}; LR = {np.exp(llr):.1f}",
        p_value=float(st.binomtest(K, N, 0.5).pvalue),
        verdict=f"data favour no cost by LR {np.exp(llr):.1f} (moderate); exact test vs 0.5 p = {st.binomtest(K, N, 0.5).pvalue:.2f}",
        evidence_type="hypothesis comparison (genuine)", predictive=False)
    cf0 = [a["cf_sham_share"] for a in sims0]
    add(id="H2", target="Chinmay courts-first share", hypothesis="observed share is a draw from the kappa = 0 model",
        test="predictive p", statistic=f"obs {K / N:.3f}; model {np.mean(cf0):.3f}", p_value=pred_p(cf0, K / N),
        verdict="", evidence_type="by construction (kappa = 0 => 0.5)", predictive=True)
    # heterogeneity across the 16 block x treatment cells
    ks = np.array([k for _, _, k, _ in C.TABLE_3_2])
    ns = np.array([n for *_, n in C.TABLE_3_2])
    phat = ks.sum() / ns.sum()
    chi_obs = float(np.sum((ks - ns * phat) ** 2 / (ns * phat * (1 - phat))))
    chi_sim = []
    for a_rng in range(R):
        rng = np.random.default_rng(SEED + 9000 + a_rng)
        kk = rng.binomial(ns, 0.5)
        ph = kk.sum() / ns.sum()
        chi_sim.append(np.sum((kk - ns * ph) ** 2 / (ns * ph * (1 - ph))))
    add(id="H3", target="Chinmay courts-first, 16 cells", hypothesis="no extra between-cell heterogeneity beyond binomial sampling",
        test="chi-square dispersion, predictive p", statistic=f"chi2 obs {chi_obs:.1f} (df 15); model median {np.median(chi_sim):.1f}",
        p_value=pred_p(chi_sim, chi_obs, "greater"), verdict="", evidence_type="model-structure check", predictive=True)
    # courtship-latency F statistics
    for term, obs_F in (("INF", 0.12546), ("FEMALES", 6.932389), ("MALE", 6.494133), ("INF*FEMALES", 0.614137),
                        ("INF*MALE", 0.006074), ("FEMALES*MALE", 0.419571)):
        simF = [a[("cl", term)][0] for a in sims0]
        mc = pred_p(simF, obs_F, "greater")
        add(id=f"H4-{term}", target=f"Chinmay latency ANOVA: {term}", hypothesis="observed F is a draw from the model (no such effect in the model)",
            test="upper tail of F(1, ~900) (model null is exact; MC check in statistic)",
            statistic=f"F obs {obs_F:.3f}; model median {np.median(simF):.3f}; MC p {mc:.3f}",
            p_value=float(st.f.sf(obs_F, 1, 900)), verdict="",
            evidence_type="model lacks genotype effects" if term in ("FEMALES", "MALE") else "by construction", predictive=True)
    # courts-most: cell means and held-out SD
    cm_obs = np.array([0.453756, 0.626823, 0.524235, 0.504487, 0.517231, 0.476364, 0.521629, 0.547363,
                       0.526507, 0.636923, 0.463822, 0.431904, 0.563251, 0.507709, 0.584606, 0.517671])
    sims_cm = np.array([[a[("cm_mean", b, t)] for b, t, _, _ in C.TABLE_3_2] for a in sims0])
    sd_cells = sims_cm.std(0, ddof=1)
    z_obs = float(np.sum(((cm_obs - 0.5) / sd_cells) ** 2))
    z_sim = np.sum(((sims_cm - 0.5) / sd_cells) ** 2, axis=1)
    add(id="H5", target="Chinmay courts-most, 16 cell means", hypothesis="cell means scatter around 0.5 as the model predicts",
        test="sum of squared standardised deviations, predictive p", statistic=f"obs {z_obs:.1f}; model median {np.median(z_sim):.1f}",
        p_value=pred_p(z_sim, z_obs, "greater"), verdict="", evidence_type="by construction (mean) + calibrated spread", predictive=True)
    sd34_obs = np.mean([0.43417, 0.394471, 0.428308, 0.458916, 0.407859, 0.457247, 0.37597, 0.424645])
    sd34_sim = [np.mean([a[("cm_sd", b, t)] for b, t, _, _ in C.TABLE_3_2 if b in (3, 4)]) for a in sims0]
    add(id="H6", target="Chinmay courts-most per-vial SD, blocks 3-4 (held out)", hypothesis="held-out SD matches the model calibrated on blocks 1-2",
        test="predictive p", statistic=f"obs {sd34_obs:.3f}; model {np.mean(sd34_sim):.3f} [{np.quantile(sd34_sim, .025):.3f}, {np.quantile(sd34_sim, .975):.3f}]",
        p_value=pred_p(sd34_sim, sd34_obs), verdict="", evidence_type="out-of-sample (within study)", predictive=True)
    # ------------------------------------------------------------ Khan
    obs_k = np.array([0.447, 0.478, 0.469, 0.459])

    def khan_exp_means(Pm, seed):
        out = []
        for r in range(R):
            k = C.simulate_khan(np.random.default_rng(seed + r), Pm, khan["modules"])
            out.extend(k["bias_scores"])
        return np.array(out)
    Ps = dict(Pk)
    Ps.update(values(load_regime("spec_a2")))
    models = {"calibrated (Byrne & Rice)": Pk, "no-choice baseline (beta 0)": dict(Pk, beta0=0.0),
              "spec-frozen beta 2.5": Ps}
    ll = {}
    for name, Pm in models.items():
        sim = khan_exp_means(Pm, SEED + 20000)
        kde = st.gaussian_kde(sim)
        ll[name] = float(np.sum(np.log(np.maximum(kde(obs_k), 1e-300))))
        add(id=f"H7-{name.split()[0]}", target="Khan Table 5 (4 experiment means)", hypothesis=f"observed experiment means are draws from: {name}",
            test="predictive p of the mean of 4", statistic=f"obs mean {obs_k.mean():.3f}; model per-experiment {sim.mean():.3f} (sd {sim.std():.3f})",
            p_value=pred_p([np.mean(np.random.default_rng(i).choice(sim, 4)) for i in range(4000)], obs_k.mean()),
            verdict="", evidence_type="out-of-sample prediction" if name.startswith("calibrated") else ("trivial baseline" if "baseline" in name else "fitted to spec"),
            predictive=True)
    # dispersion: Khan's CIs imply the between-vial SD (6 vials per experiment, t with 5 df)
    ci = [(0.412, 0.482), (0.465, 0.491), (0.419, 0.496), (0.424, 0.494)]
    sd_obs = [(hi - lo) / (2 * st.t.ppf(0.975, 5)) * np.sqrt(6) for lo, hi in ci]
    rngv = np.random.default_rng(SEED + 25000)
    b_cal = C.beta(Pk, Pk["male_depleted"])
    vial_sd = []
    for _ in range(R):
        sc = []
        for _ in range(6):
            mi, ms, *_ = C.khan_group_vial(rngv, Pk, b_cal)
            sc.append(mi / max(mi + ms, 1))
        vial_sd.append(np.std(sc, ddof=1))
    add(id="H8b", target="Khan Table 5 CIs: implied between-vial SD",
        hypothesis="vial-to-vial variation of the bias score matches the model (<= 10 matings per vial, single mating per fly)",
        test="predictive p (lower tail) of the mean implied SD",
        statistic=f"implied SD {np.mean(sd_obs):.3f} (per experiment {', '.join(f'{x:.3f}' for x in sd_obs)}); model SD {np.mean(vial_sd):.3f} [{np.quantile(vial_sd, .025):.3f}, {np.quantile(vial_sd, .975):.3f}]",
        p_value=pred_p(vial_sd, float(np.mean(sd_obs)), "less"), verdict="", evidence_type="model-structure check", predictive=True)
    add(id="H8", target="Khan Table 5", hypothesis="calibrated model vs no-choice baseline vs spec beta",
        test="log predictive density (Gaussian KDE)", statistic="; ".join(f"{k}: {v:.2f}" for k, v in ll.items()),
        p_value=np.nan, verdict=f"LR calibrated / baseline = {np.exp(ll['calibrated (Byrne & Rice)'] - ll['no-choice baseline (beta 0)']):.1f}; "
                                f"calibrated / spec = {np.exp(ll['calibrated (Byrne & Rice)'] - ll['spec-frozen beta 2.5']):.1f}",
        evidence_type="hypothesis comparison (genuine)", predictive=False)
    # ------------------------------------------------------------ Wittman & Fedorka
    Pw = dict(P, observation_min=20)
    for kap in (0.4, 1.0):
        rr = C.simulate_two_choice(np.random.default_rng(SEED + 30000), 20000, 1.0, 1 - kap, C.beta(Pw, False), Pw)
        most = rr["cm_sham"][np.isfinite(rr["cm_sham"]) & (rr["cm_sham"] != 0.5)]
        p_most = float(np.mean(most > 0.5))
        add(id=f"H9-k{kap:g}", target="Wittman & Fedorka exp. B: control courted most in 0.64 of 78 trials",
            hypothesis=f"a fecundity cue with kappa = {kap:g} (calibrated beta) explains it",
            test="exact binomial vs model probability", statistic=f"model P(courted most) = {p_most:.3f}; obs 50/78 = 0.641",
            p_value=float(st.binomtest(50, 78, p_most).pvalue), verdict="", evidence_type="out-of-sample prediction", predictive=True)
    # ------------------------------------------------------------ telegony
    tel = {}
    for name, Pm in (("HR 2 (conditioning input)", P), ("HR 1 (independent: Gupta 2016)", dict(P, sex_hazard_ratio_male=1.0))):
        res = [T.analyse_telegony(T.simulate_chinmay_telegony(np.random.default_rng(SEED + 40000 + r), Pm, F)) for r in range(R)]
        tel[name] = res
        sex = [a[("wald", "Gender")][0] for a in res]
        null_hr1 = "HR 1" in name
        add(id=f"H10-{name.split()[1]}", target="Chinmay telegony: gender Wald chi2 = 108.2", hypothesis=f"sex effect from {name}",
            test="chi2(1) upper tail (exact null)" if null_hr1 else "predictive p (upper tail; MC floor 1/(R+1))",
            statistic=f"model chi2 median {np.median(sex):.1f}, max {np.max(sex):.1f}",
            p_value=float(st.chi2.sf(108.163827, 1)) if null_hr1 else pred_p(sex, 108.163827, "greater"), verdict="",
            evidence_type="input echo" if "conditioning" in name else "out-of-sample prediction", predictive=True)
    res = tel["HR 2 (conditioning input)"]
    pop = [a[("wald", "Population")][0] for a in res]
    add(id="H11", target="Chinmay telegony: stepfather Wald chi2 = 1.02", hypothesis="no stepfather effect (compartmentalised)",
        test="chi2(2) upper tail (exact null)", statistic=f"model chi2 median {np.median(pop):.2f}; MC p {pred_p(pop, 1.02465034, 'greater'):.2f}",
        p_value=float(st.chi2.sf(1.02465034, 2)),
        verdict="", evidence_type="by construction", predictive=True)
    minlr = [min(a[("logrank", s_, t_)][2] for s_ in ("F", "M") for t_ in T.TIMEPOINTS) for a in res]
    add(id="H12", target="Chinmay telegony: smallest of 4 log-rank p = 0.047", hypothesis="arises by chance under no stepfather effect",
        test="P(min p <= 0.047) under null", statistic=f"{np.mean(np.array(minlr) <= 0.0473457):.2f}",
        p_value=float(np.mean(np.array(minlr) <= 0.0473457)), verdict="", evidence_type="generic multiple testing", predictive=True)
    # ------------------------------------------------------------ Manas: attenuation hypothesis (data only)
    rows += attenuation_test(R)
    # ------------------------------------------------------------ Manas: LH regime predictive checks
    rows += lh_predictive()
    # ------------------------------------------------------------ Manas: HRI magnitude at lambda = 1
    rows += hri_tests()
    # ------------------------------------------------------------ verdicts and FDR
    df = pd.DataFrame(rows)
    # FDR family = substantive tests only: rows that cannot fail by design
    # (by construction, input echoes, generic arithmetic) would dilute it
    trivial = df.evidence_type.astype(str).str.startswith(("by construction", "input", "generic"))
    m = df.predictive.astype(bool) & df.p_value.notna() & ~trivial
    df["in_FDR_family"] = m
    df["q_BH"] = np.nan
    df.loc[m, "q_BH"] = bh(df.loc[m, "p_value"].values)
    for i, r in df.iterrows():
        if r["verdict"]:
            continue
        if not r["predictive"] or pd.isna(r["p_value"]):
            continue
        if not r["in_FDR_family"]:
            df.at[i, "verdict"] = f"consistent (p = {r['p_value']:.2f}) [not evidence: {r['evidence_type']}]" if r["p_value"] > 0.05                 else f"p = {r['p_value']:.3g} [not evidence: {r['evidence_type']}]"
            continue
        if r["q_BH"] <= 0.05:
            base = "REJECTED (q < 0.05)"
        elif r["p_value"] < 0.05:
            base = f"TENSION (p = {r['p_value']:.3f} < 0.05, but q = {r['q_BH']:.2f} after FDR)"
        else:
            base = "consistent with model"
        if str(r["evidence_type"]).startswith(("by construction", "input")):
            base += " [not evidence: " + r["evidence_type"] + "]"
        df.at[i, "verdict"] = base
    df["informative"] = df["id"].map(INFORMATIVE).fillna("")
    df.to_csv(os.path.join(OUT, "hypothesis_tests.csv"), index=False)
    pd.set_option("display.max_colwidth", 110)
    pd.set_option("display.width", 260)
    print(df[["id", "statistic", "p_value", "q_BH", "verdict"]].to_string())


INFORMATIVE = {
    "H1": "yes, but moderate (LR 3.4)",
    "H2": "no (by construction)",
    "H7-calibrated": "weak: the no-choice baseline is also consistent (see H8)",
    "H7-no-choice": "weak",
    "H8": "weak evidence (LR < 3) for the calibrated model over the no-choice baseline",
    "H13-MF": "low power: the raw MB-FB difference is itself n.s.",
    "H13-EF": "low power",
    "H14-r_wgmf|M": "no: 95% predictive interval spans ~0.8 in r",
    "H14-r_wgmf|E": "no: 95% predictive interval spans ~0.7 in r",
    "H14-r_wgmf|F": "no: 95% predictive interval spans ~0.7 in r",
    "H15": "no: the design spans -0.8 to +0.5",
    "H16": "weak: n = 6 model replicates, paper SD approximated",
    "H19-0.001": "moderate: no collapse filter needed at lambda = 1; paper SD approximated",
    "H19-0.01": "moderate: no collapse filter needed at lambda = 1; paper SD approximated",
    "H6": "moderate (within-study held-out)",
    "H3": "yes (structure check)",
    "H8b": "yes (structure check)",
}


def attenuation_test(R):
    """H0: the true (disattenuated) intersexual correlation is equal across sex
    ratios, so the observed sex-ratio pattern is reliability attenuation only.
    Vial bootstrap within line x day cells; reliability and r recomputed in each
    resample."""
    raw = pd.read_excel(RAW, header=1)
    raw.columns = ["Sex", "SexRatio", "Day", "Line", "Fitness"]
    raw = raw.dropna(subset=["Fitness"])
    raw["Fitness"] = raw.Fitness.astype(float)

    def stats_(d):
        d = d.copy()
        d["w"] = d.Fitness / d.groupby(["Sex", "SexRatio", "Day"]).Fitness.transform("mean")
        cell = d.groupby(["Sex", "SexRatio", "Day", "Line"]).w
        cm = cell.mean().reset_index()
        lm = cm.groupby(["Sex", "SexRatio", "Line"]).w.mean().unstack([0, 1])
        within = cell.var(ddof=1).groupby(level=[0, 1]).mean()
        ncell = cell.size().groupby(level=[0, 1]).mean()
        ndays = cm.groupby(["Sex", "SexRatio", "Line"]).Day.nunique().groupby(level=[0, 1]).mean()
        out = {}
        for sr in "MEF":
            r = np.corrcoef(lm[("Female", sr)], lm[("Male", sr)])[0, 1]
            rel = []
            for sx in ("Female", "Male"):
                v = lm[(sx, sr)].var(ddof=1)
                e = within[(sx, sr)] / (ncell[(sx, sr)] * ndays[(sx, sr)])
                rel.append(max(1e-3, (v - e) / v))
            out[sr] = (r, r / np.sqrt(rel[0] * rel[1]), np.sqrt(rel[0] * rel[1]))
        return out
    obs = stats_(raw)
    rng = np.random.default_rng(SEED + 70000)
    groups = [g.index.values for _, g in raw.groupby(["Sex", "SexRatio", "Day", "Line"])]
    boots = []
    for _ in range(R):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        boots.append(stats_(raw.loc[idx]))
    rows = []
    for a, b in (("M", "F"), ("E", "F"), ("M", "E")):
        d_obs = obs[a][1] - obs[b][1]
        d_b = np.array([x[a][1] - x[b][1] for x in boots])
        # centre the bootstrap distribution at 0 to test H0: difference = 0
        p = float(np.mean(np.abs(d_b - d_b.mean()) >= abs(d_obs)))
        rows.append(dict(id=f"H13-{a}{b}", target=f"Manas BMC 2022: disattenuated r, {a} - {b}",
                         hypothesis="sex-ratio differences in r are reliability attenuation only (equal true r)",
                         test="vial bootstrap within line x day (null-centred)",
                         statistic=f"r obs {obs[a][0]:.3f} vs {obs[b][0]:.3f}; attenuation {obs[a][2]:.3f} vs {obs[b][2]:.3f}; disattenuated {obs[a][1]:.3f} vs {obs[b][1]:.3f}",
                         p_value=p, verdict="", evidence_type="data-only test of the model's mechanism", predictive=True))
    return rows


def lh_predictive():
    rows = []
    path = os.path.join(ROOT, "results", "lh_assays_default.csv")
    if not os.path.exists(path):
        return rows
    d = pd.read_csv(path)
    sh = d[d.regime == "shared"]
    for col, obs in (("r_wgmf|M", 0.3805), ("r_wgmf|E", 0.4027), ("r_wgmf|F", 0.2515),
                     ("r_male_across|M-F", 0.5567), ("r_male_across|M-E", 0.6995), ("r_male_across|F-E", 0.5415),
                     ("r_female_across|M-F", 0.7688), ("r_female_across|M-E", 0.7493), ("r_female_across|F-E", 0.8421)):
        rows.append(dict(id=f"H14-{col}", target=f"Manas BMC 2022: {col}", hypothesis="observed value is a draw from the LH 'shared' regime (6 populations x 40 assays)",
                         test="predictive p", statistic=f"obs {obs:.3f}; model {sh[col].mean():.3f} [{sh[col].quantile(.025):.3f}, {sh[col].quantile(.975):.3f}]",
                         p_value=pred_p(sh[col].values, obs), verdict="",
                         evidence_type="calibrated (variances from same data)" if "across" in col else "regime choice (ours)", predictive=True))
    des = os.path.join(ROOT, "results", "design_runs_default.csv")
    if os.path.exists(des):
        dd = pd.read_csv(des)
        rows.append(dict(id="H15", target="Manas BMC 2022: r(E) = 0.40", hypothesis="prior predictive over the architecture design (12 points x 2 populations)",
                         test="predictive p over the design", statistic=f"design r_assay_E range {dd.r_assay_E.min():.2f} to {dd.r_assay_E.max():.2f}; median {dd.r_assay_E.median():.2f}",
                         p_value=pred_p(dd.r_assay_E.values, 0.4027), verdict="", evidence_type="prior predictive (architecture unknown)", predictive=True))
    return rows


def hri_tests():
    rows = []
    path = os.path.join(ROOT, "results", "lambda_sensitivity.csv")
    if not os.path.exists(path):
        return rows
    d = pd.read_csv(path)
    l1 = d[(d.lam == 1) & np.isclose(d.map_length_M, 0.1)]
    if len(l1) >= 3:
        # paper: mean -0.078 over 50 replicates; replicate range -0.30 to 0.125 => SD approximated as range/4
        sd_paper = (0.125 + 0.30) / 4
        t = st.ttest_ind_from_stats(l1.r_mf.mean(), l1.r_mf.std(ddof=1), len(l1), -0.0780, sd_paper, 50, equal_var=False)
        rows.append(dict(id="H16", target="HRI preprint: mean r_mf,W = -0.078 at 0.1 M", hypothesis="model (lambda = 1, paper's own parameters) has the same mean",
                         test="Welch t (paper SD approximated from the reported range)", statistic=f"model {l1.r_mf.mean():+.3f} (sd {l1.r_mf.std(ddof=1):.3f}, n={len(l1)}); paper -0.078 (n=50)",
                         p_value=float(t.pvalue), verdict="", evidence_type="replication of the same model", predictive=True))
        t0 = st.ttest_1samp(l1.r_mf, 0.0)
        rows.append(dict(id="H17", target="HRI preprint: r_mf,W < 0 at 0.1 M", hypothesis="model mean differs from the trivial baseline 0",
                         test="one-sample t (lambda = 1 replicates)", statistic=f"t = {t0.statistic:.2f}", p_value=float(t0.pvalue),
                         verdict="underpowered: with n = 6 the model mean is not distinguishable from 0, so H16 cannot discriminate either",
                         evidence_type="power check of the replication", predictive=False))
    # paper medians read from Fig. 2 (docs/04 4.5); the short-map replicate SD is not
    # reported, so the 0.1 M range-based SD is reused (an approximation, stated in the row)
    paper_median = {0.001: -0.115, 0.01: -0.125}
    for R_ in (0.001, 0.01):
        sub = d[(d.lam == 1) & np.isclose(d.map_length_M, R_)]
        if len(sub):
            coll = int(np.sum(sub.V_f > 0.1))
            rows.append(dict(id=f"H18-{R_:g}", target=f"HRI collapse filter at {R_:g} M", hypothesis="collapse is a rescaling artefact (absent at lambda = 1)",
                             test="count of collapsed replicates at lambda = 1", statistic=f"{coll}/{len(sub)} collapsed; mean r (all) {sub.r_mf.mean():+.3f}",
                             p_value=np.nan, verdict="artefact confirmed" if coll == 0 else "collapse also at full scale: filter NOT justified",
                             evidence_type="audit validation", predictive=False))
            if len(sub) >= 3:
                sd_paper = (0.125 + 0.30) / 4
                t = st.ttest_ind_from_stats(sub.r_mf.mean(), sub.r_mf.std(ddof=1), len(sub), paper_median[R_], sd_paper, 50, equal_var=False)
                rows.append(dict(id=f"H19-{R_:g}", target=f"HRI preprint: median r_mf,W = {paper_median[R_]} at {R_:g} M (Fig. 2)",
                                 hypothesis="model (lambda = 1, paper's own parameters, no filter) has the same centre",
                                 test="Welch t (paper SD approximated by the 0.1 M range/4)",
                                 statistic=f"model {sub.r_mf.mean():+.3f} (sd {sub.r_mf.std(ddof=1):.3f}, n={len(sub)}); paper {paper_median[R_]:+.3f}",
                                 p_value=float(t.pvalue), verdict="", evidence_type="replication of the same model", predictive=True))
    return rows


if __name__ == "__main__":
    main()

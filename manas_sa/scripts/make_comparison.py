"""Build the comparison table (paper | model | sign / order / magnitude |
driving assumption), the coverage table, and the four required plots, from
whatever result files exist in results/.

usage: python scripts/make_comparison.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = os.path.join(ROOT, "results")
INK, C1, C2, C3, MUTED = "#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SRN = {"M": "male-biased", "E": "equal", "F": "female-biased"}

# [paper] values (claim type A) -------------------------------------------------
PAPER_BMC = {"r_wgmf": {"M": (0.3805, 0.2992, 0.5283), "E": (0.4027, 0.3140, 0.5526), "F": (0.2515, 0.1198, 0.4502)},
             "prop_SA": {"M": (0.3097, 0.2358, 0.3504), "E": (0.2986, 0.2237, 0.3430), "F": (0.3742, 0.2749, 0.4401)}}
PAPER_ACROSS = {("r_female_across", "M-F"): 0.7688, ("r_female_across", "M-E"): 0.7493, ("r_female_across", "F-E"): 0.8421,
                ("r_male_across", "M-F"): 0.5567, ("r_male_across", "M-E"): 0.6995, ("r_male_across", "F-E"): 0.5415}
# HRI Figure 2 medians, read from the figure (A, approximate, +-0.01)
PAPER_HRI_MEDIAN = {0.001: -0.115, 0.01: -0.125, 0.1: -0.07, 1.0: -0.02, 10.0: 0.0}
PAPER_HRI_MEAN_01 = -0.0780     # stated in text at 0.1 M, mu = 7e-7
CONTRAST = [("Chippindale 2001 (LHM, 40)", -0.30, None, None), ("Innocenti & Morrow 2010 (LHM, 100)", -0.52, -0.86, -0.10),
            ("Collet 2016 LHM-UU (100)", -0.41, -0.41 - 1.96 * 0.18, -0.41 + 1.96 * 0.18),
            ("Collet 2016 LHM-UCL (113)", 0.21, 0.21 - 1.96 * 0.19, 0.21 + 1.96 * 0.19),
            ("Ruzicka 2019 (LHM, 223)", 0.15, -0.21, 0.46)]


def save(fig, name):
    """savefig via a temp file + os.replace, retried: on Windows a freshly
    written PNG can be briefly locked (viewer / virus scan) -> Errno 22."""
    import time
    path = os.path.join(RES, name)
    for attempt in range(6):
        try:
            tmp = path + f".tmp{attempt}.png"
            fig.savefig(tmp)
            os.replace(tmp, path)
            break
        except OSError:
            time.sleep(1.0 + attempt)
    plt.close(fig)


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def sign_word(a, b):
    return "matches sign" if np.sign(a) == np.sign(b) else "misses sign"


def recompute_word(rc, published):
    if np.isnan(rc):
        return ""
    return "reproduced (|diff| < 0.02)" if abs(rc - published) < 0.02 else f"recomputed differs by {rc - published:+.3f}"


def mag_word(model, lo, hi, label="paper CI"):
    if lo is None:
        return "magnitude not scoreable"
    return "within " + label if lo <= model <= hi else ("misses magnitude (model above)" if model > hi else "misses magnitude (model below)")


def build_table():
    """Comparison table from the result files, with no side effects.
    Returns (table, inputs needed by the plots)."""
    rows = []
    # ---------------- target 1 / thesis Ch 3-4: recomputed from raw data (B) and LH-model predictions
    rec = load("bmc2022_recomputed.csv")
    lha = None
    for prof in ("full", "default", "quick"):
        lha = load(f"lh_assays_{prof}.csv")
        if lha is not None:
            break
    for stat in ("r_wgmf", "prop_SA"):
        for s in ("M", "E", "F"):
            est, lo, hi = PAPER_BMC[stat][s]
            row = dict(target="1 BMC 2022 (Table 2A)", statistic=f"{stat} {SRN[s]}", paper=est, paper_CI=f"[{lo}, {hi}]")
            if rec is not None:
                rr = rec[(rec.statistic == stat) & (rec.level == s)]
                if len(rr):
                    row["recomputed_from_raw_data"] = round(float(rr.recomputed.iloc[0]), 4)
                    row["verdict_recompute"] = recompute_word(float(rr.recomputed.iloc[0]), est)
            if lha is not None:
                for reg in lha.regime.unique():
                    v = lha[lha.regime == reg][f"{stat}|{s}"]
                    row[f"model_{reg}"] = f"{v.mean():.3f} [{v.quantile(0.025):.3f}, {v.quantile(0.975):.3f}]"
                v = lha[lha.regime == "shared"][f"{stat}|{s}"] if "shared" in set(lha.regime) else None
                if v is not None:
                    row["verdict_shared"] = f"{sign_word(v.mean(), est)}; {mag_word(v.mean(), lo, hi)}"
                    if stat == "prop_SA":
                        row["verdict_shared"] += " [identity: prop_SA = (1 - r)/2, no information beyond r]"
            row["driving_assumption"] = "sexually concordant (shared deleterious) variation dominates; assay noise calibrated to variances only"
            rows.append(row)
    # ordering across sex ratios
    if lha is not None and "shared" in set(lha.regime):
        sh = lha[lha.regime == "shared"]
        p_order = "r(F) lowest; M ~ E; difference n.s."
        frac_F_lowest = float(np.mean((sh["r_wgmf|F"] < sh["r_wgmf|M"]) & (sh["r_wgmf|F"] < sh["r_wgmf|E"])))
        diff = sh["r_wgmf|M"] - sh["r_wgmf|F"]
        rows.append(dict(target="1 BMC 2022", statistic="ordering of r_wgmf across sex ratios", paper=p_order,
                         paper_CI="MB-FB: [-0.0721, 0.3507]",
                         model_shared=f"P(F lowest) = {frac_F_lowest:.2f}; MB-FB mean {diff.mean():.3f} [{diff.quantile(.025):.3f}, {diff.quantile(.975):.3f}]",
                         verdict_shared=(("matches ordering" if diff.mean() > 0 else "misses ordering") + "; trend not significant in model assays either" if diff.quantile(.025) < 0 else "matches ordering")
                         + " [echo of calibration: data-only attenuation (no simulator) gives the same ordering, E 0.837 > M 0.809 > F 0.721; explains ~1/3 of the FB drop; hypothesis test H13: equal disattenuated r not rejected, p = 0.30-0.34, low power]",
                         driving_assumption="beta_s(SR) calibrated to line variances (lower genetic signal-to-noise at FB) -> attenuation"))
        for key, val in PAPER_ACROSS.items():
            col = f"{key[0]}|{key[1]}"
            if col in sh:
                v = sh[col]
                rr = rec[(rec.statistic == key[0]) & (rec.level == key[1])] if rec is not None else []
                rc = float(rr.recomputed.iloc[0]) if len(rr) else np.nan
                rows.append(dict(target="1 BMC 2022 (Table 2A)", statistic=f"{key[0]} {key[1]}", paper=val,
                                 recomputed_from_raw_data=round(rc, 4), verdict_recompute=recompute_word(rc, val),
                                 model_shared=f"{v.mean():.3f} [{v.quantile(.025):.3f}, {v.quantile(.975):.3f}]",
                                 verdict_shared=f"{sign_word(v.mean(), val)}; " + ("within model 95% range" if v.quantile(.025) <= val <= v.quantile(.975) else "outside model 95% range"),
                                 driving_assumption="no true genotype x sex-ratio interaction in the model; correlations < 1 arise from sampling noise only"))
        for sex in ("female", "male"):
            for s in ("M", "E", "F"):
                col = f"cross_over_within_{sex}|{s}"
                if col in sh and rec is not None:
                    rr = rec[(rec.statistic == f"cross_over_within_{sex}") & (rec.level == s)]
                    emp = float(rr.recomputed.iloc[0]) if len(rr) else np.nan
                    v = sh[col]
                    rows.append(dict(target="2 thesis Ch.4 / 3 AmNat (substitute)", statistic=f"cross/within-sex covariance ratio, {sex}, {SRN[s]}",
                                     paper="(not reported; computed from raw data)", recomputed_from_raw_data=round(emp, 3),
                                     model_shared=f"{v.mean():.3f} [{v.quantile(.025):.3f}, {v.quantile(.975):.3f}]",
                                     verdict_shared=f"{sign_word(v.mean(), emp)}; " + ("within model 95% range" if v.quantile(.025) <= emp <= v.quantile(.975) else "outside model 95% range"),
                                     driving_assumption="Robertson/FTNS decomposition of fitness itself; trait-level AmNat estimates not accessible"))
    # ---------------- target 4: HRI
    hri = None
    hri_all = None
    for prof in ("full", "default", "quick"):
        hri = load(f"hri_runs_{prof}.csv")
        if hri is not None:
            break
    if hri is not None:
        # degenerate replicates (neo-Y-like haplotype classes; a rescaling artefact:
        # none occur at lambda = 1) are counted and reported, then excluded
        hri_all = hri.copy()
        hri["collapsed"] = hri.V_f / hri.lam ** 2 > 0.1
        for (lab, R), g in hri.groupby(["label", "map_length_M"]):
            if g.collapsed.any():
                rows.append(dict(target="4 (rescaling check)", statistic=f"{lab} at {R} M: collapsed replicates",
                                 paper="V ~ 0.02 in all replicates", model_HRI=f"{int(g.collapsed.sum())}/{len(g)} collapsed "
                                 f"(V_f up to {g.V_f.max():.0f}, r = -1); excluded from the rows above",
                                 verdict_HRI="rescaling artefact (lambda = 2); none at lambda = 1",
                                 driving_assumption="Muller's-ratchet speed and N*V_g not invariant under rescaling"))
        hri = hri[~hri.collapsed]
        mp = hri[hri.label == "maplen/paper"]
        for R, med in PAPER_HRI_MEDIAN.items():
            d = mp[np.isclose(mp.map_length_M, R)]
            if len(d) == 0:
                continue
            m, se = d.r_mf.mean(), d.r_mf.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(target="4 HRI bioRxiv 2025 (Fig. 2)", statistic=f"r_mf,W at map length {R} M (both sexes recombine)",
                             paper=f"median ~{med} (read from figure)", model_shared=None,
                             model_HRI=f"mean {m:+.3f} (se {se:.3f}), median {d.r_mf.median():+.3f}, n={len(d)}, lambda={int(d.lam.iloc[0])}",
                             verdict_HRI=("matches sign" if (med == 0 and abs(m) < 2 * se + 0.03) or np.sign(m) == np.sign(med) else "misses sign")
                             + ("; magnitude within 0.05" if abs(d.r_mf.median() - med) < 0.05 else "; misses magnitude"),
                             driving_assumption="purely sex-limited selection + linkage; rescaled population (see lambda sensitivity)"))
        # trend with map length
        g = mp.groupby("map_length_M").r_mf.mean()
        if len(g) >= 3:
            rho = pd.Series(np.log10(g.index)).corr(pd.Series(g.values), method="spearman")
            rows.append(dict(target="4 HRI bioRxiv 2025", statistic="trend: bias weakens with map length", paper="monotone weakening; absent at 10 M",
                             model_HRI=f"Spearman(log map, mean r) = {rho:+.2f}; mean r at max map = {g.iloc[-1]:+.3f}",
                             verdict_HRI="matches ordering" if rho > 0.7 else "partly matches ordering",
                             driving_assumption="HRI LD decays with recombination"))
        dro = hri[hri.label == "maplen/drosophila"]
        for R in sorted(dro.map_length_M.unique()):
            d = dro[np.isclose(dro.map_length_M, R)]
            dp = mp[np.isclose(mp.map_length_M, R)]
            rows.append(dict(target="4 (extension, C)", statistic=f"r_mf,W at {R} M, no male crossing over",
                             paper="(not simulated in paper)", model_HRI=f"mean {d.r_mf.mean():+.3f} vs paper-setting {dp.r_mf.mean():+.3f}",
                             verdict_HRI="prediction (C)", driving_assumption="Drosophila males do not recombine"))
        mu = pd.concat([hri[hri.label == "mu_sweep/paper"],
                        mp[np.isclose(mp.map_length_M, 0.1) & np.isclose(mp.mu_bp, 7e-7)]])
        if mu.mu_bp.nunique() >= 3:
            g = mu.groupby("mu_bp").r_mf.mean()
            rho = pd.Series(g.index).corr(pd.Series(g.values), method="spearman")
            rows.append(dict(target="4 HRI bioRxiv 2025 (Fig. 3)", statistic="trend: bias strengthens with mutation rate",
                             paper="more negative at higher mu", model_HRI=f"Spearman(mu, mean r) = {rho:+.2f}",
                             verdict_HRI="matches ordering" if rho < -0.7 else ("partly matches ordering" if rho < 0 else "misses ordering"),
                             driving_assumption="more segregating sites -> more HRI"))
        d01 = mp[np.isclose(mp.map_length_M, 0.1)]
        if len(d01):
            lam_ = float(d01.lam.iloc[0])
            vf, vm = d01.V_f.mean() / lam_ ** 2, d01.V_m.mean() / lam_ ** 2
            rows.append(dict(target="4 HRI bioRxiv 2025", statistic="V_f, V_m at 0.1 M (rescaled back by 1/lambda^2)",
                             paper="~0.02", model_HRI=f"V_f {vf:.3f}, V_m {vm:.3f} (raw {d01.V_f.mean():.3f}, lambda={lam_:g})",
                             verdict_HRI="matches magnitude" if 0.01 <= vf <= 0.04 else "misses magnitude",
                             driving_assumption="V_g scales as lambda^2 under N*s-preserving rescaling"))
            lo_map = mp[mp.map_length_M <= 0.01].interference_pct.median()
            hi_map = mp[np.isclose(mp.map_length_M, 10)].interference_pct.median()
            rows.append(dict(target="4 HRI bioRxiv 2025", statistic="median selective interference (COV/V_f x 100): <=0.01 M vs 10 M",
                             paper="~10% at low map length; ~0 at 10 M",
                             model_HRI=f"{lo_map:.1f}% vs {hi_map:.1f}%",
                             verdict_HRI="matches sign; " + ("magnitude within 2x" if 5 <= -lo_map <= 20 else "misses magnitude"),
                             driving_assumption="negative COV = interference (paper reports its size)"))
    rows += extra_rows()
    # ---------------- target 5: modifier
    mcl = load("modifier_claims.csv")
    if mcl is not None:
        for _, r in mcl.iterrows():
            rows.append(dict(target="5 modifier bioRxiv 2023", statistic=r.claim, paper=r.reported, model_modifier=r.model,
                             verdict_modifier=r.verdict, driving_assumption="authors' own recursion code re-executed (exact grid)"))
    lam = load("lambda_sensitivity.csv")
    if lam is not None:
        lam = lam.assign(collapsed=lam.V_f / lam.lam ** 2 > 0.1)
        for (R, L), g in lam.groupby(["map_length_M", "lam"]):
            ncol = int(g.collapsed.sum())
            verdict = "rescaling diagnostic"
            if L == 1:
                verdict = (f"paper's own scale, no filter; {ncol}/{len(g)} collapsed"
                           + (" -> lambda = 2 collapse is a rescaling artefact (filter justified; H18/H19)" if ncol == 0 else ""))
            rows.append(dict(target="4 (rescaling check)", statistic=f"r_mf,W at {R} M, lambda={int(L)}",
                             paper=f"median ~{PAPER_HRI_MEDIAN.get(R)}",
                             model_HRI=f"mean {g.r_mf.mean():+.3f} (se {g.r_mf.std(ddof=1) / np.sqrt(len(g)):.3f}, n={len(g)})",
                             verdict_HRI=verdict, driving_assumption="N*V_g is not invariant under Nmu/Nr/Ns rescaling"))
    l4 = load("hri_runs_lambda4_partial.csv")
    if l4 is not None:
        for R, d in l4.groupby("map_length_M"):
            rows.append(dict(target="4 (rescaling check)", statistic=f"r_mf,W at {R} M, lambda=4",
                             paper=f"median ~{PAPER_HRI_MEDIAN.get(R)}", model_HRI=f"mean {d.r_mf.mean():+.3f} (n={len(d)})",
                             verdict_HRI="rescaling artefact: neo-Y-like haplotype classes",
                             driving_assumption="N*V_g and Muller's-ratchet rate not invariant under rescaling"))
    return pd.DataFrame(rows), (rec, lha, hri, hri_all)


def main():
    comp, (rec, lha, hri, hri_all) = build_table()
    comp.to_csv(os.path.join(RES, "comparison_table.csv"), index=False)
    coverage(comp)
    print(comp.to_string(max_colwidth=60))
    plots(rec, lha, hri)
    plot_distinguish(hri_all)


def first_existing(stem):
    for prof in ("full", "default", "quick"):
        d = load(f"{stem}_{prof}.csv")
        if d is not None:
            return d, prof
    return None, None


def extra_rows():
    """Null regimes, power, emulator-based contrast reconciliation and the
    distinguishing experiment."""
    rows = []
    lhp, _ = first_existing("lh_assays")
    if lhp is not None:
        for reg, expect in (("shared", "r > 0 (concordant variation dominates)"),
                            ("sexlimited", "r ~ 0 (no SA, no shared effects; only LD)"), ("trueSA", "r < 0")):
            d = lhp[lhp.regime == reg]
            if len(d) == 0:
                continue
            v = d["r_wgmf|E"]
            ht = d.groupby("pop").r_hemigenome_true.first()
            if reg == "sexlimited":
                verdict = "null holds" if abs(v.mean()) < 0.1 else "null fails"
            elif reg == "trueSA":
                verdict = "null holds" if v.mean() < 0 else "null fails"
            else:
                verdict = "as designed" if v.mean() > 0 else "fails"
            rows.append(dict(target="null / regime check", statistic=f"LH regime '{reg}': r_w,g,mf (equal SR); true hemigenome r",
                             paper=expect,
                             model_shared=f"assay {v.mean():+.3f} [{v.quantile(.025):+.3f}, {v.quantile(.975):+.3f}]; "
                                          f"true {ht.mean():+.3f} (populations {ht.min():+.2f} to {ht.max():+.2f})",
                             verdict_shared=verdict, driving_assumption="mutation-class mixture (config [lh])"))
    for stem, arch_name in (("power_sexratio", "shared regime, r~0.1"), ("power_sexratio_lhlike", "LH-like architecture, r~0.4")):
        pw, _ = first_existing(stem)
        if pw is None:
            continue
        for n, g in pw.groupby("n_lines"):
            p = g.significant_positive.mean()
            rows.append(dict(target="1 BMC 2022 (design, C)", statistic=f"power ({arch_name}): MB - FB difference in r_w,g,mf, {n} lines",
                             paper="difference n.s. with 39 lines" if n == 39 else "(not tested)",
                             model_shared=f"P(95% CI excludes 0) = {p:.2f}; P(FB lowest) = {g.F_lowest.mean():.2f}; "
                                          f"mean difference {g.diff_MB_FB.mean():+.3f}",
                             verdict_shared=("matches (a n.s. result is the expected outcome)" if p < 0.5 else "misses")
                             if n == 39 else "design prediction (C)",
                             driving_assumption="attenuation-only model of the sex-ratio effect"))
    des, prof = first_existing("design_runs")
    if des is not None:
        lam_d = float(des.lam.iloc[0])
        # degenerate (partially neo-Y-like) populations: genetic variance far above
        # the stable regimes (V_f/lambda^2 ~ 0.025) and pooled-sex r_pop ~ -1
        des["collapsed"] = (des.V_f / lam_d ** 2 > 0.1) | (des.r_pop < -0.8)
        for k, g in des.groupby("point"):
            rows.append(dict(target="contrast reconciliation (C, IBM design)",
                             statistic=f"design point {k}: f_SA={g.f_SA.iloc[0]:.2f}, concordant share={g.sc_share.iloc[0]:.2f}, "
                                       f"female map={10 ** g.log10_R.iloc[0]:.2f} M",
                             paper="LH +0.40; LHM -0.52 to +0.21",
                             model_shared=f"39-line assay r {g.r_assay_E.mean():+.3f} (populations: "
                                          + ", ".join(f"{v:+.2f}" for v in g.r_assay_E) + f"); hemigenome r {g.r_hemi_true.mean():+.3f}; "
                                          f"pooled-sex r_pop {g.r_pop.mean():+.3f}; collapsed {int(g.collapsed.sum())}/{len(g)}",
                             verdict_shared="reaches LH value (selected from the sweep = fitted, not predicted)" if g.r_assay_E.mean() > 0.3 else
                             ("reaches LHM negative range" if g.r_assay_E.mean() < -0.25 else "intermediate"),
                             driving_assumption="mutation-class mixture; replicate populations differ by history alone"))
        c = des[des.collapsed]
        if len(c):
            rows.append(dict(target="method check (C)", statistic="pooled-sex genotype r_mf,W vs hemigenome r in collapsed populations",
                             paper="(estimator of target 4 pools female and male genotypes)",
                             model_shared=f"{len(c)}/{len(des)} populations collapsed; r_pop {c.r_pop.mean():+.2f} vs hemigenome r {c.r_hemi_true.mean():+.2f}",
                             verdict_shared="estimator confounded by intersexual genetic differentiation (C)",
                             driving_assumption="female-detrimental haplotypes carried only by males (no male recombination, SA benefits)"))
    val, _ = first_existing("emulator_validation")
    validated = False
    if val is not None:
        v = val[val.target == "r_assay_E"].iloc[0]
        validated = v.GP_cv_rmse < 1.5 * v.replicate_sd
        rows.append(dict(target="emulator (Layer 3)", statistic="grouped-CV RMSE of GP / GBM vs replicate-population sd (assay r)",
                         paper="-", model_shared=f"GP {v.GP_cv_rmse:.3f}, GBM {v.GBM_cv_rmse:.3f}, replicate sd {v.replicate_sd:.3f}",
                         verdict_shared="validated" if validated else "not validated: emulator inversion not identified",
                         driving_assumption="12 design points x 2 populations; large between-population variance"))
    inv, _ = first_existing("emulator_inversion")
    if inv is not None and validated:
        for _, r in inv.iterrows():
            if r.P_as_extreme_by_sampling > 0.05:
                verdict = "sampling alone suffices"
            else:
                verdict = "needs more SA than the LH-like architecture" if r.published_r < r.emulator_r_at_shared_regime \
                    else "needs less SA than the LH-like architecture"
            rows.append(dict(target="contrast reconciliation (C)", statistic=f"{r.study}: published r = {r.published_r:+.2f}",
                             paper=r.published_r,
                             model_shared=f"LH-like architecture predicts {r.emulator_r_at_shared_regime:+.3f}; "
                                          f"P(as extreme by sampling) = {r.P_as_extreme_by_sampling:.3f}; "
                                          f"f_SA consistent: [{r.f_SA_consistent_min:.2f}, {r.f_SA_consistent_max:.2f}]",
                             verdict_shared=verdict,
                             driving_assumption="fraction of SA mutations f_SA (GP emulator of the IBM design sweep)"))
    ne = load("ne_check.csv")
    lam1 = load("lambda_sensitivity.csv")
    if ne is not None:
        g = ne.groupby("N").r_mf.agg(["mean", "std", "count"])
        if lam1 is not None and (lam1.lam == 1).any():
            l1 = lam1[(lam1.lam == 1) & np.isclose(lam1.map_length_M, 0.1)].r_mf
            g.loc[2500] = [l1.mean(), l1.std(), l1.count()]
        g = g.sort_index()
        sd = float(ne.r_mf.std(ddof=1))
        delta = float(g["mean"].iloc[:-1].mean() - g["mean"].iloc[-1])
        n_req = int(np.ceil(2 * (1.96 + 0.84) ** 2 * sd ** 2 / delta ** 2)) if delta < 0 else -1
        rows.append(dict(target="distinguishing experiment (C)", statistic="N-dependence of r_mf,W under sex-limited selection (0.1 M, lambda=1 per-generation parameters)",
                         paper="(proposed experiment: HRI predicts more negative r at small N)",
                         model_shared="; ".join(f"N={int(n)}: {r['mean']:+.3f} (sd {r['std']:.3f}, n={int(r['count'])})" for n, r in g.iterrows())
                                      + (f"; populations per N for 80% power ~{n_req}" if n_req > 0 else ""),
                         verdict_shared="direction as HRI predicts, not monotone; weak (C)" if delta < 0 else "no N effect",
                         driving_assumption="HRI ~ 1/(N c); replicate-population sd limits power"))
    dist, prof = first_existing("distinguish")
    if dist is not None:
        dpw = load(f"distinguish_power_{prof}.csv")
        for name, g in dist.groupby("population"):
            m = g.groupby("k").r.mean()
            pstr = ""
            if dpw is not None and len(dpw):
                q = dpw[dpw.population == name].groupby("n_lines").power_two_sided.mean()
                pstr = "; power (k=4): " + ", ".join(f"n={int(n)}: {v:.2f}" for n, v in q.items())
            rows.append(dict(target="distinguishing experiment (C)", statistic=f"{name}: r after k generations of female meiosis",
                             paper="(proposed experiment)",
                             model_shared=f"r0 {m.get(0, np.nan):+.3f}, k=4 {m.get(4, np.nan):+.3f}, "
                                          f"k=16 {m.get(16, np.nan):+.3f}, LE {g.r_LE.mean():+.3f}{pstr}",
                             verdict_shared="prediction (C)",
                             driving_assumption="SA = direct term survives recombination; HRI/LD = indirect term decays"))
    return rows


NOT_SCOREABLE = {
    "1": "MCMCglmm h2 and r (Table 2B, not re-fitted); daily random-effect variances; LMM p-values",
    "2": "trait-level genetic correlations and selection gradients (trait data not deposited; MB gradients garbled, F4)",
    "3": "everything: Am Nat 2025 not accessible (HTTP 403, no preprint); no number is quoted",
    "4": "per-map-length and per-mu values beyond medians read from boxplots",
    "5": "heatmap cell values (no deposited numbers, F11)",
}


def coverage(comp):
    rows = []
    vcols = [c for c in comp.columns if c.startswith("verdict")]
    for key, why in NOT_SCOREABLE.items():
        sub = comp[comp.target.astype(str).str.startswith(key)]
        v = sub[vcols].fillna("").astype(str).agg(" ".join, axis=1) if len(sub) else pd.Series(dtype=str)
        rows.append(dict(target=key, statistics_scored=len(sub),
                         matches=int(v.str.contains("matches|within|holds|suffices").sum()),
                         misses=int(v.str.contains("misses|outside|fails|does not").sum()),
                         not_scoreable=why))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "coverage.csv"), index=False)


def plot_distinguish(hri_all):
    """r after k generations of female meiosis for six representative
    populations (collapsed HRI replicates removed), with the linkage-equilibrium
    value (all LD removed) at the right."""
    dist, prof = first_existing("distinguish")
    if dist is None:
        return
    if hri_all is not None:
        h = hri_all[hri_all.label.str.startswith("maplen/")].copy()
        h["population"] = "hri_" + h.label.str.split("/").str[1] + "_R" + h.map_length_M.map(lambda v: f"{v:g}")
        bad = set(zip(h[h.V_f / h.lam ** 2 > 0.1].population, h[h.V_f / h.lam ** 2 > 0.1].rep))
        dist = dist[[(p, r) not in bad for p, r in zip(dist.population, dist.rep)]]
    show = [("lh_shared", "LH 'shared' (mostly concordant)", C1), ("lh_trueSA", "LH true SA (collapsed)", C3),
            ("lh_sexlimited", "LH sex-limited null", C2), ("hri_paper_R0.01", "HRI, 0.01 M", "#7c5cbf"),
            ("hri_paper_R0.1", "HRI, 0.1 M", "#c2417c"), ("hri_paper_R1", "HRI, 1 M", MUTED)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (name, lab, col) in enumerate(show):
        g = dist[dist.population == name]
        if g.empty:
            continue
        m = g.groupby("k").r.mean()
        se = g.groupby("k").r.std(ddof=1) / np.sqrt(g.groupby("k").r.count())
        xs = np.array(m.index) + 0.5
        ax.errorbar(xs, m.values, yerr=1.96 * se.values, fmt="o-", ms=4, lw=2, capsize=2, color=col,
                    label=f"{lab} (n={g.rep.nunique()})")
        ax.plot([32 * 1.07 ** i], [g.r_LE.mean()], "D", ms=7, color=col, markeredgecolor="white")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.axvline(25, color="#c3c2b7", lw=0.8, ls=":")
    ax.text(38, ax.get_ylim()[1], "all LD\nremoved", ha="center", va="top", fontsize=7, color=MUTED)
    ax.set(xscale="log", xlabel="generations of female meiosis among hemigenomes (+0.5)",
           ylabel="r between female and male hemigenome values",
           title="Recombining hemigenomes in the lab does not break HRI-type LD")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    save(fig, f"fig_distinguish_{prof}.png")


def plots(rec, lha, hri):
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})
    # 1. r vs recombination
    if hri is not None:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for lab, col, name in (("maplen/paper", C1, "model: both sexes recombine (paper setting)"),
                               ("maplen/drosophila", C2, "model: no male crossing over (Drosophila)")):
            d = hri[hri.label == lab]
            if len(d):
                g = d.groupby("map_length_M").r_mf
                ax.errorbar(g.mean().index, g.mean(), yerr=1.96 * g.std(ddof=1) / np.sqrt(g.count()), fmt="o-", color=col,
                            capsize=3, label=name)
        Rs = sorted(PAPER_HRI_MEDIAN)
        ax.plot(Rs, [PAPER_HRI_MEDIAN[r] for r in Rs], "s--", color=INK, label="paper: median read from Fig. 2")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xscale="log", xlabel="map length of the simulated region (Morgans)", ylabel="r_mf,W",
               title="Sex-limited selection + linkage: r_mf,W vs recombination")
        ax.legend(fontsize=8)
        save(fig, "fig_r_vs_recombination.png")
        ld = None
        for prof in ("full", "default", "quick"):
            p = os.path.join(RES, f"hri_ld_{prof}.csv")
            if os.path.exists(p):
                ld = pd.read_csv(p)
                break
        if ld is not None and len(ld):
            fig, ax = plt.subplots(figsize=(7, 4.4))
            for (R, mr), d in ld.groupby(["map_length_M", "male_recombination"]):
                g = d.groupby("mid").mean_D.mean()
                ax.plot(g.index, g.values, "o-", ms=4, label=f"{R} M, male rec={bool(mr)}")
            ax.axhline(0, color=MUTED, lw=1)
            ax.set(xscale="log", xlabel="map distance between a female-limited and a male-limited site (M)",
                   ylabel="mean D (derived alleles; same sign as beneficial alleles)",
                   title="LD vs map distance under purely sex-limited selection")
            ax.legend(fontsize=7, ncol=2)
            save(fig, "fig_ld_vs_distance.png")
    # 2. r vs sex ratio, 4. Robertson within vs cross
    if lha is not None:
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
        x = np.arange(3)
        ax = axs[0]
        est = [PAPER_BMC["r_wgmf"][s][0] for s in "MEF"]
        lo = [PAPER_BMC["r_wgmf"][s][1] for s in "MEF"]
        hi = [PAPER_BMC["r_wgmf"][s][2] for s in "MEF"]
        ax.errorbar(x - 0.15, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="s", color=INK, capsize=3,
                    label="paper (Table 2A, 95% CI)")
        for k, (reg, col) in enumerate((("shared", C1), ("sexlimited", C2), ("trueSA", C3))):
            d = lha[lha.regime == reg]
            if len(d) == 0:
                continue
            m = [d[f"r_wgmf|{s}"].mean() for s in "MEF"]
            q1 = [d[f"r_wgmf|{s}"].quantile(.025) for s in "MEF"]
            q2 = [d[f"r_wgmf|{s}"].quantile(.975) for s in "MEF"]
            ax.errorbar(x + 0.05 + 0.1 * k, m, yerr=[np.subtract(m, q1), np.subtract(q2, m)], fmt="o", color=col,
                        capsize=3, label=f"model: {reg} (39-line assays, 95% range)")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x, ["male-biased", "equal", "female-biased"])
        ax.set(ylabel="r_w,g,mf (line means)", title="Intersexual correlation for fitness vs adult sex ratio")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        ax = axs[1]
        if rec is not None:
            for sex, mk in (("female", "s"), ("male", "D")):
                emp = [float(rec[(rec.statistic == f"cross_over_within_{sex}") & (rec.level == s)].recomputed.iloc[0]) for s in "MEF"]
                ax.plot(x - 0.1, emp, mk + "--", color=INK, label=f"data (recomputed): {sex}")
        d = lha[lha.regime == "shared"]
        if len(d):
            for sex, mk, col in (("female", "s", C1), ("male", "D", C2)):
                m = [d[f"cross_over_within_{sex}|{s}"].mean() for s in "MEF"]
                ax.plot(x + 0.1, m, mk + "-", color=col, label=f"model (shared): {sex}")
        ax.set_xticks(x, ["male-biased", "equal", "female-biased"])
        ax.set(ylabel="cross-sex / within-sex covariance of relative fitness",
               title="Robertson/FTNS decomposition: cross- vs within-sex response")
        ax.legend(fontsize=7)
        save(fig, "fig_r_vs_sexratio_and_robertson.png")


if __name__ == "__main__":
    main()

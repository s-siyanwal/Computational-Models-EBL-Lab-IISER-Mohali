"""Coupled model run (docs/09_coupling.md).

Stage 1 (genetic layer, iasc/ibm/regime.py)
  - BRB-like ancestor: burn-in with the unchanged Manas engine at lambda = 2;
  - per scenario and block: I and U populations of 150 + 150 sampled from the
    ancestor, T generations of the IUS regime at lambda = 1, male recombination 0;
  - writes results/coupled/generations.csv (record summary every generation)
    and results/coupled/records.csv (per-individual record of the evolved and
    ancestral populations).
Stage 2 (assay layer through extensions/coupler.py; reads the records only)
  - Chinmay Ch. 3 two-choice design on the evolved I and U genotypes;
  - Khan & Prasad (2013) group assay with male betas from the ancestor record;
  - writes outputs/coupled_comparison.csv with evidence_type on every row.

usage: python scripts/run_coupled.py [--stage evolve|assay|all] [--reps 200]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from iasc import params as IP  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402
from iasc.ibm import regime as Rg  # noqa: E402

OUTD = os.path.join(ROOT, "results", "coupled")
SEED = 20261006


def load_cfg():
    with open(os.path.join(ROOT, "config", "coupling.toml"), "rb") as f:
        return tomllib.load(f)


def environment(cc, scen):
    e = cc["environment"]
    env = Rg.Environment(H1=e["H1"], nu=e["nu"], t_lay=e["t_lay"], target_survival=e["target_survival"],
                         sex_factor_f=e["sex_factor_f"], sex_factor_m=e["sex_factor_m"],
                         assay_dose=e["assay_dose"], assay_hours=e["assay_hours"], cost_decay=scen["cost_decay"])
    # a0 such that the ancestral genotype (clear_rel = tol_rel = 1) has kappa = kappa_anc_96h at 96 h, OD 1.5
    env.a0 = -np.log(1.0 - scen["kappa_anc_96h"]) / (e["assay_dose"] ** e["nu"] * np.exp(-scen["cost_decay"] * 96.0))
    return env


def build_architecture(cc, core):
    rng = np.random.default_rng(SEED)
    a = cc["architecture"]
    lh = core["lh"]
    arch = A.build(rng, a["sites_A"], a["sites_X"], mix=lh[a["mix"]], gamma_shape=lh["gamma_shape"],
                   gamma_scale=lh["gamma_scale"], sa_benefit_ratio=lh["sa_benefit_ratio"])
    tarch = Rg.add_traits(rng, arch, a["trait_sites_A"], a["trait_sites_X"], {t: a["trait_sd"] for t in Rg.TRAITS})
    return arch, tarch


def evolve(cc, core, tag=""):
    os.makedirs(OUTD, exist_ok=True)
    arch, tarch = build_architecture(cc, core)
    lh = core["lh"]
    b = cc["burnin"]
    lam = b["lam"]
    N_b = int(round(b["N_ref"] / lam))
    gens_b = int(round(b["generations_ref"] / lam))
    assert core["drosophila"]["male_recombination"] is False
    anc_path = os.path.join(OUTD, "ancestor.npz")
    if os.path.exists(anc_path):
        G_anc = np.load(anc_path)["genomes"]
    else:
        t0 = time.time()
        G_anc = np.zeros((2 * N_b, arch.WA + arch.WX), np.uint64)
        E.evolve_one(N_b, gens_b, arch.WA, arch.WX, arch.posA, arch.posX, lh["R_A_M"] * lam, lh["R_X_M"] * lam,
                     lh["mu_bp"] * lh["L_bp_A"] * lam, lh["mu_bp"] * lh["L_bp_X"] * lam,
                     arch.eff_f * lam, arch.eff_m * lam, lh["noise_sd"], False, SEED + 1, G_anc)
        np.savez_compressed(anc_path, genomes=G_anc, lam=lam, N=N_b, generations=gens_b)
        print(f"burn-in N={N_b} gens={gens_b} lambda={lam}: {time.time() - t0:.0f}s", flush=True)
    # neutral burn-in: the ancestor's genomes do not depend on the trait effect sizes, so the
    # effects are scaled to give the stated ancestral SD exactly (finite-sites saturation makes
    # the infinite-sites formula behind trait_sd overestimate it; docs/09 9.6)
    target = cc["architecture"].get("trait_sd_ancestor")
    scale = {}
    for t in Rg.TRAITS:
        bv = Rg.trait_bv(G_anc, N_b, arch, tarch, t)
        sel = {"both": slice(None), "F": slice(0, N_b // 2), "M": slice(N_b // 2, N_b)}[Rg.EXPRESSED_IN[t]]
        sd0 = float(np.std(bv[sel]))
        scale[t] = target / sd0 if target else 1.0
        tarch.eff[t] = tarch.eff[t] * scale[t]
    print("trait effect scale (ancestral SD -> target):", {t: round(v, 3) for t, v in scale.items()}, flush=True)
    anc = Rg.ancestor_reference(G_anc, arch, tarch)
    rates = Rg.Rates(RA=lh["R_A_M"], RX=lh["R_X_M"], UA=lh["mu_bp"] * lh["L_bp_A"], UX=lh["mu_bp"] * lh["L_bp_X"],
                     noise_sd=lh["noise_sd"], male_recombination=core["drosophila"]["male_recombination"])
    s = cc["selection"]
    gens, recs = [], []
    env0 = environment(cc, cc["scenario"][0])
    recs.append(Rg.genotype_record(G_anc, arch, tarch, env0, anc, {"scenario": "ancestor", "block": 0,
                                                                   "population": "ancestor", "generation": 0},
                                   eff_scale=lam))
    for si, scen in enumerate(cc["scenario"]):
        env = environment(cc, scen)
        for blk in range(1, s["blocks"] + 1):
            for ri, reg in enumerate(("I", "U")):
                rng = np.random.default_rng(SEED + 1000 * si + 10 * blk + ri)
                G0 = Rg.sample(G_anc, N_b, s["n_per_sex"], s["n_per_sex"], rng)
                lab = {"scenario": scen["name"], "block": blk, "population": reg}
                t0 = time.time()
                G = Rg.run_regime(G0, arch, tarch, rates, reg, s["generations"], env, anc, SEED + 100 * si + 10 * blk + ri,
                                  rng, label=lab, on_generation=gens.append)
                recs.append(Rg.genotype_record(G, arch, tarch, env, anc, dict(lab, generation=s["generations"])))
                print(f"[{scen['name']}] block {blk} {reg}: {time.time() - t0:.1f}s; "
                      f"clear_rel {gens[-1]['clear_rel_mean']:.3f} kappa13 {gens[-1]['kappa_13h_mean']:.3f}", flush=True)
    pd.DataFrame(gens).to_csv(os.path.join(OUTD, f"{tag}generations.csv"), index=False)
    pd.concat(recs).to_csv(os.path.join(OUTD, f"{tag}records.csv"), index=False)
    print("wrote", OUTD)


# --------------------------------------------------------------------------- assay stage
CM_OBS = np.array([0.453756, 0.626823, 0.524235, 0.504487, 0.517231, 0.476364, 0.521629, 0.547363,
                   0.526507, 0.636923, 0.463822, 0.431904, 0.563251, 0.507709, 0.584606, 0.517671])
CL_F_OBS = {"INF": 0.12546, "INF*FEMALES": 0.614137, "INF*MALE": 0.006074}
KHAN_MEAN_OBS = 0.463
KHAN_VIAL_SD_OBS = 0.029


def pred_p(sim, obs, side="two"):
    sim = np.asarray(sim, float)
    sim = sim[np.isfinite(sim)]
    lo = (np.sum(sim <= obs) + 1) / (len(sim) + 1)
    hi = (np.sum(sim >= obs) + 1) / (len(sim) + 1)
    return {"two": min(1.0, 2 * min(lo, hi)), "lower": lo, "greater": hi}[side]


def obs_chinmay():
    from extensions import choice as C
    from extensions import stats as S
    t = pd.DataFrame(C.TABLE_3_2, columns=["block", "trt", "K", "N"])
    o = {"cf_share": t.K.sum() / t.N.sum(),
         "cf_I_fem": t[t.trt.str[0] == "I"].K.sum() / t[t.trt.str[0] == "I"].N.sum(),
         "cf_U_fem": t[t.trt.str[0] == "U"].K.sum() / t[t.trt.str[0] == "U"].N.sum(),
         "cf_I_male": t[t.trt.str[-1] == "I"].K.sum() / t[t.trt.str[-1] == "I"].N.sum(),
         "cf_U_male": t[t.trt.str[-1] == "U"].K.sum() / t[t.trt.str[-1] == "U"].N.sum(),
         "cm_mean_cells": float(CM_OBS.mean())}
    o["cf_fem_diff"] = o["cf_U_fem"] - o["cf_I_fem"]
    o["cf_male_diff"] = o["cf_U_male"] - o["cf_I_male"]
    g = t.groupby("block")[["K", "N"]].sum()
    o["all_blocks_ns"] = bool(all(S.binom_two_sided(int(k), int(n)) > 0.05 for k, n in g.values))
    return o


def chinmay_stats(recs, a):
    from extensions import choice as C
    d = pd.DataFrame(recs)
    d = d[d.courted]
    f = lambda m: float(d[m].cf_sham.mean())  # noqa: E731
    out = dict(cf_share=float(d.cf_sham.mean()), cf_I_fem=f(d.female == "I"), cf_U_fem=f(d.female == "U"),
               cf_I_male=f(d.male == "I"), cf_U_male=f(d.male == "U"),
               all_blocks_ns=all(a[("cf_p_block", b)] > 0.05 for b in (1, 2, 3, 4)),
               cm_mean_cells=float(np.mean([a[("cm_mean", b, t)] for b, t, _, _ in C.TABLE_3_2])))
    out["cf_fem_diff"] = out["cf_U_fem"] - out["cf_I_fem"]
    out["cf_male_diff"] = out["cf_U_male"] - out["cf_I_male"]
    for term in CL_F_OBS:
        out[f"F_{term}"] = a[("cl", term)][0]
    return out


def assay(cc, reps, tag=""):
    from extensions import choice as C
    from extensions import coupler as K
    from extensions.config import load_regime, values
    ius, khan = load_regime("ius"), load_regime("khan")
    P = values(ius)
    Pk = dict(P)
    Pk.update(values(khan))
    F = dict(ius["modules"], coupling=True)
    Fk = dict(khan["modules"], coupling=True)
    rec = pd.read_csv(os.path.join(OUTD, f"{tag}records.csv"))
    obs = obs_chinmay()
    rows = []
    near0_k, near0_b = 0.05, 0.1                          # [ours] "near 0" thresholds
    anc = rec[rec.population == "ancestor"]
    for si, scen in enumerate(cc["scenario"]):
        r = rec[rec.scenario == scen["name"]]
        env = environment(cc, scen)
        k_anc = float(Rg.kappa(1.0, 1.0, env, env.assay_dose, env.assay_hours))
        kI = r[(r.population == "I") & (r.sex == "F")].kappa_13h.mean()
        kU = r[(r.population == "U") & (r.sex == "F")].kappa_13h.mean()
        bI = r[(r.population == "I") & (r.sex == "M")].beta_rel.mean()
        bU = r[(r.population == "U") & (r.sex == "M")].beta_rel.mean()
        sims = []
        for k in range(reps):
            recs = K.simulate_chinmay_study(np.random.default_rng(SEED + 7 * k + si), P, F, r,
                                            np.random.default_rng(SEED + 100000 + 7 * k + si))
            sims.append(chinmay_stats(recs, C.analyse_chinmay(recs)))
        sim = pd.DataFrame(sims)
        by_hand = scen["kappa_anc_96h"] == 0.0
        et = ("by construction (kappa = 0 set by hand for every genotype)" if by_hand
              else "prediction, prior-dependent (kappa_anc_96h and cost_decay are MISSING scenario values)")
        leak = "no target value used; beta from Byrne & Rice; genetic layer never reads assay outcomes"
        base = dict(scenario=scen["name"], kappa_anc_96h=scen["kappa_anc_96h"], cost_decay_per_h=scen["cost_decay"])

        def add(row_id, target, paper, model, p, evidence, verdict, note=""):
            rows.append(dict(base, id=row_id, target=target, paper_estimate=paper, model_estimate=model,
                             p_value=p, evidence_type=evidence, verdict=verdict, leakage_note=note or leak))

        def verdict_of(p):
            return "consistent" if p >= 0.05 else "TENSION/REJECTED (p < 0.05)"

        # genotype-level rows (genetic layer outputs)
        add("G-kappa13", "kappa(G, OD 1.5, 13 h): evolved I vs U females", "MISSING (not measured at 12-14 h)",
            f"I {kI:.3f}, U {kU:.3f}; ancestor {k_anc:.3f}", np.nan,
            "by construction" if by_hand else "derived from genetic layer (prior-dependent)",
            "U keeps the ancestral value (no infection selection in U)")
        add("G-beta", "male preference beta_rel(G): evolved I vs U", "MISSING",
            f"I {bI:.3f}, U {bU:.3f} (calibration mean = 1)", np.nan, "derived (neutral in both regimes)",
            "near 0: no" if min(bI, bU) > near0_b else "near 0: yes")
        sI = np.exp(-r[r.population == "I"].H96_assay).mean()
        sU = np.exp(-r[r.population == "U"].H96_assay).mean()
        add("G-survival", "survival advantage of I over U at OD 1.5 (96 h)", "0.35-0.40 (Gupta 2016 p.82, p.95)",
            f"{sI - sU:+.3f} (I {sI:.3f}, U {sU:.3f})", np.nan,
            "existential / prior-dependent (mutational variance and T are MISSING; not fitted)",
            "sign as observed" if sI > sU else "wrong sign",
            "U survival 0.2 at OD 1.5 is an input (dose exponent derived from it)")
        lI = (1 / r[(r.population == "I") & (r.sex == "F")].clear_rel).mean()
        lU = (1 / r[(r.population == "U") & (r.sex == "F")].clear_rel).mean()
        add("G-load", "pathogen load of I relative to U females (load ~ 1/clearance)", "about 0.7 (Gupta 2016 p.95: ~30% fewer CFUs, I vs S)",
            f"{lI / lU:.3f}", np.nan, "existential / prior-dependent", "direction as observed" if lI < lU else "wrong direction")
        # A1 rows
        for key, target, side in (("cf_share", "A1 courts-first share (sham), pooled", "two"),
                                  ("cm_mean_cells", "A1 courts-most, mean of 16 cell means", "two")):
            p = pred_p(sim[key], obs[key], side)
            add(f"A1-{key}", target, f"{obs[key]:.3f}",
                f"{sim[key].mean():.3f} [{sim[key].quantile(.025):.3f}, {sim[key].quantile(.975):.3f}]", p, et,
                verdict_of(p) + (" [not evidence: by construction]" if by_hand else ""))
        pall = float(sim.all_blocks_ns.mean())
        add("A1-cf-blocks", "A1 all four per-block courts-first tests n.s. (two-sided)", str(obs["all_blocks_ns"]),
            f"P(all n.s.) = {pall:.2f}", pall, et, "consistent" if pall >= 0.05 else "TENSION/REJECTED")
        pF = pred_p(sim["F_INF"], CL_F_OBS["INF"], "two")
        add("A1-CL-INF", "A1 latency ANOVA infection term, F", "F = 0.125, p = 0.723",
            f"median F {sim['F_INF'].median():.2f}", pF, et, verdict_of(pF) + (" [not evidence: by construction]" if by_hand else ""))
        explained = (kI <= near0_k and kU <= near0_k and k_anc > near0_k) or (min(bI, bU) <= near0_b)
        add("A1-explained", "A1 explained by IUS evolution (kappa(13 h) or beta near 0 produced by evolution, then assay consistent)",
            "null in all 16 cells", f"kappa_I {kI:.3f}, kappa_U {kU:.3f}, ancestor {k_anc:.3f}; beta_I {bI:.2f}, beta_U {bU:.2f}",
            np.nan, et,
            "YES" if explained else ("NO: kappa = 0 is hand-set (by construction)" if by_hand else
                                     "NO: U is not selected on kappa or beta, so U keeps the ancestral kappa(13 h); a null in U x U vials cannot come from IUS evolution"))
        # new derived predictions
        p = pred_p(sim.cf_fem_diff, obs["cf_fem_diff"])
        add("NEW-fem-x-inf", "female genotype x infection in courts-first: share(U females) - share(I females)",
            f"{obs['cf_fem_diff']:+.3f}", f"{sim.cf_fem_diff.mean():+.3f} [{sim.cf_fem_diff.quantile(.025):+.3f}, {sim.cf_fem_diff.quantile(.975):+.3f}]",
            p, "by construction (0)" if by_hand else "new derived prediction (sign derived from I selection on clearance; magnitude prior-dependent)",
            verdict_of(p))
        p = pred_p(sim.cf_male_diff, obs["cf_male_diff"])
        add("NEW-male-x-inf", "male genotype x infection in courts-first: share(U males) - share(I males)",
            f"{obs['cf_male_diff']:+.3f}", f"{sim.cf_male_diff.mean():+.3f} [{sim.cf_male_diff.quantile(.025):+.3f}, {sim.cf_male_diff.quantile(.975):+.3f}]",
            p, "derived null (preference neutral in both regimes; drift only)", verdict_of(p))
        for term in ("INF*FEMALES", "INF*MALE"):
            pF = pred_p(sim[f"F_{term}"], CL_F_OBS[term], "two")
            add(f"NEW-CL-{term}", f"latency ANOVA {term} (derived, not hand-added)", f"F = {CL_F_OBS[term]}",
                f"median F {sim[f'F_{term}'].median():.2f}", pF,
                "by construction" if by_hand else "derived (via evolved kappa / beta)", verdict_of(pF))
        print(f"[{scen['name']}] CF {sim.cf_share.mean():.3f}; fem diff {sim.cf_fem_diff.mean():+.3f}", flush=True)
    # ---------------- Khan A2 with genetic variance in beta (ancestor males)
    kr = {}
    for lab, rr in (("beta genetic", anc), ("beta fixed", anc.assign(beta_rel=1.0))):
        means, sds = [], []
        for k in range(reps):
            o = K.simulate_khan(np.random.default_rng(SEED + 31 * k), Pk, Fk, rr, np.random.default_rng(SEED + 7777 + k))
            means.append(np.mean(o["bias_scores"]))
            sds.append(np.mean([np.std(v, ddof=1) for v in o["vial_scores"] if len(v) > 1]))
        kr[lab] = (np.array(means), np.array(sds))
    base = dict(scenario="Khan regime", kappa_anc_96h=np.nan, cost_decay_per_h=np.nan)
    m, s = kr["beta genetic"]
    m0, s0 = kr["beta fixed"]
    pm = pred_p(m, KHAN_MEAN_OBS)
    rows.append(dict(base, id="A2-khan-bias", target="Khan Table 5 mean bias (infected mated / all)", paper_estimate="0.463",
                     model_estimate=f"{m.mean():.3f} [{np.quantile(m, .025):.3f}, {np.quantile(m, .975):.3f}] (beta fixed: {m0.mean():.3f})",
                     p_value=pm, evidence_type="out-of-sample (beta from Byrne & Rice; genotype variation from the ancestor record; kappa 0.4 measured)",
                     verdict="consistent" if pm >= 0.05 else "TENSION/REJECTED",
                     leakage_note="beta never retuned on Khan Table 5"))
    ps = pred_p(s, KHAN_VIAL_SD_OBS, "lower")
    rows.append(dict(base, id="H8b-coupled", target="Khan between-vial SD implied by the CIs", paper_estimate="0.029",
                     model_estimate=f"{np.nanmean(s):.3f} (beta fixed: {np.nanmean(s0):.3f})", p_value=ps,
                     evidence_type="model-structure check", verdict="REJECTED (stays): genetic variance in beta cannot reduce vial-to-vial variance" if ps < 0.05 else "consistent",
                     leakage_note="not hidden in the genetic layer; needs a mating-scheme or CI-interpretation fix"))
    # ---------------- rows that the coupling does not change
    static = [
        ("SA-courtship", "SA locus with no kappa/beta/cue effect changes CF, CL, CM?", "-", "no: bit-identical statistics (tests/test_coupling.py)",
         "by construction (regression test)", "holds"),
        ("telegony", "stepfather genotype (incl. SA) shifts offspring survival?", "no (thesis Table 4.1-4.2: population n.s.)",
         "no: coupler passes nothing to telegony.py; compartmentalisation on", "by construction", "consistent [not evidence]"),
        ("H10-coupled", "telegony sex chi2 = 108", "chi2 = 108.16", "record sex factor = 1 (no independent data; Gupta 2016 gives none)",
         "input absent", "REJECTED (stays)"),
        ("H4-coupled", "latency male / female genotype main effects", "F = 6.49 / 6.93", "no latency trait in the genetic layer: intercepts not identified",
         "not identified", "TENSION (stays)"),
        ("H14-coupled", "female across-sex-ratio correlations", "0.749-0.842", "no genotype x sex-ratio term added", "not implemented", "TENSION (stays)"),
        ("male-recomb", "male recombination in Drosophila runs", "0", "0 in every Drosophila run; True only in paper-setting replications (SLiM default)",
         "audit", "no rerun needed; H19 unchanged"),
    ]
    for i, t, pe, me, et, v in static:
        rows.append(dict(scenario="all", id=i, target=t, paper_estimate=pe, model_estimate=me, p_value=np.nan,
                         evidence_type=et, verdict=v, leakage_note=""))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", f"coupled_comparison{'_' + tag.rstrip('_') if tag else ''}.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(out[["scenario", "id", "paper_estimate", "model_estimate", "p_value", "verdict"]].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("evolve", "assay", "all"))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--tag", default="", help="file prefix for records and outputs, e.g. sens_sd_realised_")
    ap.add_argument("--no-sd-scaling", action="store_true", help="sensitivity: keep the raw trait_sd effects")
    args = ap.parse_args()
    IP.below_normal_priority()
    cc, core = load_cfg(), IP.load()
    if args.stage in ("evolve", "all"):
        if args.no_sd_scaling:
            cc["architecture"]["trait_sd_ancestor"] = None
        evolve(cc, core, args.tag)
    if args.stage in ("assay", "all"):
        assay(cc, args.reps, args.tag)


if __name__ == "__main__":
    main()

"""Layer 2, regime (i): LH-like laboratory populations and the hemiclone assay.

1. Evolve LH-like populations (X + autosome, no male crossing over, N=1920
   rescaled) under three genetic architectures:
     shared     - mostly sexually concordant (SC) deleterious variation, some
                  sex-limited and some SA mutations
     sexlimited - purely sex-limited (null: no SA, no shared effects)
     trueSA     - SA-dominated variation
2. For each population compute the population r_mf,W (genotype-based, as in
   the HRI preprint), then emulate the hemiclonal assay: calibrate the
   sex-ratio multipliers beta_s(SR) and noise to the empirical line variances
   and within-cell CV^2 ONLY, then run repeated 39-line assays and apply the
   paper's estimators (r_w,g,mf, SA proportion, across-sex-ratio r, cross/within).

usage: python scripts/run_lh.py --profile quick|default|full
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from iasc import estimators as Es  # noqa: E402
from iasc import params as P  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402
from iasc.ibm import hemiclone as H  # noqa: E402
from iasc.ibm import stats as St  # noqa: E402

RAW = os.path.join(os.path.dirname(ROOT), "experimental_papers_manas", "12862_2022_1992_MOESM3_ESM.xlsx")
REGIMES = {"shared": "mix_shared", "sexlimited": "mix_sexlimited", "trueSA": "mix_trueSA"}


def empirical_targets():
    raw = pd.read_excel(RAW, header=1)
    raw.columns = ["Sex", "SexRatio", "Day", "Line", "Fitness"]
    raw = raw.dropna(subset=["Fitness"])
    raw["Fitness"] = raw.Fitness.astype(float)
    raw["Line"] = raw.Line.astype(int)
    return Es.all_stats(raw)


def calibrate(bv_cal, acfg, targets, seed):
    """Per sex x sex ratio: find (beta, noise sd) so that the simulated line
    variance of relative fitness and within-cell CV^2 match the empirical
    values (common random numbers, averaged over 4 assay draws)."""
    beta, noise = {}, {"day": 0.10}
    for sex in ("Female", "Male"):
        for sr in Es.SR:
            tv = targets[(f"linevar_rel_{sex.lower()}", sr)]
            tc = targets[(f"within_cv2_{sex.lower()}", sr)]

            def resid(th):
                b, sd = np.exp(th)
                vals_v, vals_c = [], []
                for k in range(4):
                    rng = np.random.default_rng(seed + k)
                    d = H.run_assay(bv_cal, acfg, {(sex, sr): b}, {(sex, sr): sd, "day": noise["day"]}, rng,
                                    sexes=(sex,), srs=(sr,))
                    d = d.assign(Fitness=d.Fitness)
                    per_day = d.groupby(["Day", "Line"]).Fitness.mean().reset_index()
                    lm = per_day.groupby("Line").Fitness.mean()
                    vals_v.append((lm / lm.mean()).var(ddof=1))
                    g = d.groupby(["Day", "Line"]).Fitness
                    vals_c.append(np.nanmean(g.var(ddof=1) / g.mean() ** 2))
                return [np.log(np.mean(vals_v) / tv), np.log(np.mean(vals_c) / tc)]

            sol = least_squares(resid, x0=np.log([0.3, 0.4]), bounds=(np.log([1e-3, 1e-3]), np.log([5.0, 5.0])),
                                diff_step=0.05, max_nfev=60)
            b, sd = np.exp(sol.x)
            beta[(sex, sr)] = b
            noise[(sex, sr)] = sd
    return beta, noise


def save_populations(path, genomes, arch, rs):
    """Evolved genomes + architecture, so that the design, power and contrast
    analyses re-sample the same populations without re-evolving them."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, genomes=genomes, posA=arch.posA, posX=arch.posX, eff_f=arch.eff_f, eff_m=arch.eff_m,
                        kind=arch.kind, WA=arch.WA, WX=arch.WX, N=rs.N, lam=rs.lam)


def load_populations(path):
    z = np.load(path)
    arch = A.Architecture(z["posA"], z["posX"], z["eff_f"], z["eff_m"], z["kind"], int(z["WA"]), int(z["WX"]))
    return z["genomes"], arch


def instrument(profile, regime="shared"):
    """The calibrated assay 'instrument': median beta and noise per sex x sex
    ratio over the populations of one regime (calibrated to variances only)."""
    cal = pd.read_csv(os.path.join(ROOT, "results", f"lh_calibration_{profile}.csv"))
    cal = cal[cal.regime == regime].groupby(["sex", "sex_ratio"])[["beta", "noise_sd"]].median()
    beta = {k: float(v) for k, v in cal.beta.items()}
    noise = {k: float(v) for k, v in cal.noise_sd.items()}
    noise["day"] = 0.10
    return beta, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", choices=["quick", "default", "full"])
    ap.add_argument("--regimes", default="shared,sexlimited,trueSA")
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    lh, acfg = cfg["lh"], cfg["assay"]
    lam = cfg["rescale"]["lam_" + args.profile]
    n_pop = {"quick": 2, "default": 6, "full": 12}[args.profile]
    n_assay = {"quick": 10, "default": 40, "full": 100}[args.profile]
    targets = empirical_targets()
    rs = A.rescale(lh["N"], lh["generations_ref"], lh["mu_bp"], lh["L_bp_A"], lh["R_A_M"], lh["gamma_scale"], lam,
                   L_bp_X=lh["L_bp_X"], R_X_ref=lh["R_X_M"])
    print("rescaled:", rs, flush=True)
    rows_pop, rows_assay, rows_cal = [], [], []
    seed = cfg["meta"]["master_seed"] + 7
    for reg in [r for r in args.regimes.split(",") if r]:
        rng = np.random.default_rng(seed)
        arch = A.build(rng, n_sites_A=cfg["rescale"]["sites_autosome"], n_sites_X=cfg["rescale"]["sites_X"],
                       mix=lh[REGIMES[reg]], gamma_shape=lh["gamma_shape"], gamma_scale=rs.gamma_scale,
                       sa_benefit_ratio=lh["sa_benefit_ratio"])
        W = arch.WA + arch.WX
        out = np.zeros((n_pop, 2 * rs.N, W), np.uint64)
        seeds = np.arange(n_pop, dtype=np.int64) + seed
        t0 = time.time()
        E.evolve_many(rs.N, rs.n_gen, arch.WA, arch.WX, arch.posA, arch.posX, rs.R_A, rs.R_X, rs.U_A, rs.U_X,
                      arch.eff_f, arch.eff_m, lh["noise_sd"], cfg["drosophila"]["male_recombination"], seeds, out)
        print(f"[{reg}] evolved {n_pop} populations in {time.time() - t0:.0f}s", flush=True)
        save_populations(os.path.join(ROOT, "results", "pops", f"lh_{reg}_{args.profile}.npz"), out, arch, rs)
        for p in range(n_pop):
            st = St.population_stats(out[p], arch, rng, autosome_only=True)
            st.update(regime=reg, pop=p)
            rows_pop.append(st)
            bv_cal = H.hemigenome_bv(out[p], arch, rng, n_lines=200, n_background=4000)
            beta, noise = calibrate(bv_cal, acfg, targets, seed=1000 + p)
            for (sex, sr), b in beta.items():
                rows_cal.append(dict(regime=reg, pop=p, sex=sex, sex_ratio=sr, beta=b, noise_sd=noise[(sex, sr)]))
            # true genetic correlation of hemigenome breeding values (assay scale, no noise)
            tr = np.corrcoef(bv_cal["h_f"], bv_cal["h_mA"] + bv_cal["h_mX"])[0, 1]
            for a in range(n_assay):
                arng = np.random.default_rng(seed * 7 + p * 1000 + a)
                bv = H.hemigenome_bv(out[p], arch, arng, n_lines=acfg["n_lines"], n_background=4000)
                d = H.run_assay(bv, acfg, beta, noise, arng)
                s = Es.all_stats(d)
                rec = {f"{k[0]}|{k[1]}": v for k, v in s.items()}
                rec.update(regime=reg, pop=p, assay=a, r_hemigenome_true=tr, r_pop=st["r_mf"])
                rows_assay.append(rec)
            print(f"  [{reg}] pop {p}: r_pop={st['r_mf']:+.3f} r_hemi_true={tr:+.3f}  "
                  f"assay r(E)={np.mean([x['r_wgmf|E'] for x in rows_assay if x['regime'] == reg and x['pop'] == p]):+.3f}",
                  flush=True)
            pd.DataFrame(rows_pop).to_csv(os.path.join(ROOT, "results", f"lh_populations_{args.profile}.csv"), index=False)
            pd.DataFrame(rows_assay).to_csv(os.path.join(ROOT, "results", f"lh_assays_{args.profile}.csv"), index=False)
            pd.DataFrame(rows_cal).to_csv(os.path.join(ROOT, "results", f"lh_calibration_{args.profile}.csv"), index=False)
        seed += 100
    print("done")


if __name__ == "__main__":
    main()

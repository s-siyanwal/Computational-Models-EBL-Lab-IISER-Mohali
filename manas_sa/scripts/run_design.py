"""Design sweep (training data for the Layer-3 emulator), contrast-set
sampling and power analysis. Everything here re-uses ONE fixed assay
"instrument": the beta_s(SR) / noise values calibrated (to variances only) on
the 'shared' LH regime by scripts/run_lh.py.

Part A  design: Latin hypercube over the genetic architecture of LH-like
        populations (X + autosome, no male crossing over)
          f_SA      fraction of new mutations that are sexually antagonistic, in [0, 0.2]
                    (above ~0.2 populations collapse into neo-Y-like haplotype
                    classes with r = -1; results/design_runs_default_fSA0-0.5_collapsed.csv)
          sc_share  fraction of the remaining (non-SA) mutations that are
                    sexually concordant (rest: sex-limited, half each sex)
          log10_R   female map length of the autosome (Morgans)
        -> r_pop (genotypes), r_hemi_true (hemigenome breeding values),
           assay r_w,g,mf at 39 lines in each sex ratio, and the sampling
           distribution of r at the contrast studies' numbers of genomes.
Part B  power: on the saved 'shared' populations, the probability that the
        male-biased minus female-biased difference in r_w,g,mf is declared
        significant (95% line-bootstrap CI excludes 0) vs number of lines.

usage: python scripts/run_design.py --profile default --points 12 --reps 4
       python scripts/run_design.py --profile default --power_only
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
from scipy.stats import qmc  # noqa: E402

from iasc import estimators as Es  # noqa: E402
from iasc import params as P  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402
from iasc.ibm import hemiclone as H  # noqa: E402
from iasc.ibm import stats as St  # noqa: E402
from scripts.run_lh import instrument, load_populations  # noqa: E402

# numbers of genomes in the contrast studies (docs/01_source_map.md)
CONTRAST_N = {"Chippindale2001": 40, "Innocenti2010": 100, "Collet2016_UCL": 113, "Ruzicka2019": 223}
POWER_LINES = (39, 80, 160, 320)


def mix_of(f_sa, sc_share):
    rest = 1.0 - f_sa
    return {1: rest * (1 - sc_share) / 2, 2: rest * (1 - sc_share) / 2, 3: rest * sc_share, 4: f_sa / 2, 5: f_sa / 2}


def assay_r(genomes, arch, acfg, beta, noise, rng, n_lines, srs=Es.SR):
    bv = H.hemigenome_bv(genomes, arch, rng, n_lines=n_lines, n_background=4000)
    d = H.run_assay(bv, acfg, beta, noise, rng, srs=srs)
    return Es.paper_stats(Es.line_means(d)), d


def part_a(cfg, args, beta, noise):
    lh, acfg = cfg["lh"], cfg["assay"]
    lam = cfg["rescale"]["lam_" + args.profile]
    design = qmc.LatinHypercube(d=3, seed=cfg["meta"]["master_seed"]).random(args.points)
    design = qmc.scale(design, [0.0, 0.0, -1.0], [0.2, 1.0, np.log10(2.0)])
    out_csv = os.path.join(ROOT, "results", f"design_runs_{args.profile}.csv")
    rows = []
    for k, (f_sa, sc, lr) in enumerate(design):
        R_A = 10 ** lr
        rs = A.rescale(lh["N"], lh["generations_ref"], lh["mu_bp"], lh["L_bp_A"], R_A, lh["gamma_scale"], lam,
                       L_bp_X=lh["L_bp_X"], R_X_ref=lh["R_X_M"])
        seed = cfg["meta"]["master_seed"] + 300000 + 1000 * k
        rng = np.random.default_rng(seed)
        arch = A.build(rng, n_sites_A=cfg["rescale"]["sites_autosome"], n_sites_X=cfg["rescale"]["sites_X"],
                       mix=mix_of(f_sa, sc), gamma_shape=lh["gamma_shape"], gamma_scale=rs.gamma_scale,
                       sa_benefit_ratio=lh["sa_benefit_ratio"])
        out = np.zeros((args.reps, 2 * rs.N, arch.WA + arch.WX), np.uint64)
        t0 = time.time()
        E.evolve_many(rs.N, rs.n_gen, arch.WA, arch.WX, arch.posA, arch.posX, rs.R_A, rs.R_X, rs.U_A, rs.U_X,
                      arch.eff_f, arch.eff_m, lh["noise_sd"], cfg["drosophila"]["male_recombination"],
                      np.arange(args.reps, dtype=np.int64) + seed, out)
        for r in range(args.reps):
            st = St.population_stats(out[r], arch, rng, autosome_only=False)
            bv = H.hemigenome_bv(out[r], arch, rng, n_lines=min(300, int(1.4 * rs.N)), n_background=4000)
            rec = dict(point=k, rep=r, f_SA=f_sa, sc_share=sc, log10_R=lr, lam=lam, N=rs.N,
                       r_pop=st["r_mf"], V_f=st["V_f"], V_m=st["V_m"], n_seg=st["n_seg"],
                       r_hemi_true=float(np.corrcoef(bv["h_f"], bv["h_mA"] + bv["h_mX"])[0, 1]))
            arng = np.random.default_rng(seed + 17 * (r + 1))
            acc = {s: [] for s in Es.SR}
            for _ in range(args.n_assay):
                ps, _ = assay_r(out[r], arch, acfg, beta, noise, arng, acfg["n_lines"])
                for s in Es.SR:
                    acc[s].append(ps[("r_wgmf", s)])
            for s in Es.SR:
                rec[f"r_assay_{s}"] = float(np.mean(acc[s]))
                rec[f"r_assay_{s}_sd"] = float(np.std(acc[s], ddof=1))
            rec["P_r_assay_E_negative"] = float(np.mean(np.array(acc["E"]) < 0))
            for name, n in CONTRAST_N.items():
                v = [assay_r(out[r], arch, acfg, beta, noise, arng, n, srs=("E",))[0][("r_wgmf", "E")]
                     for _ in range(max(5, args.n_assay // 2))]
                rec[f"r_E_n{n}_mean"] = float(np.mean(v))
                rec[f"r_E_n{n}_sd"] = float(np.std(v, ddof=1))
            rows.append(rec)
        print(f"point {k}: f_SA={f_sa:.2f} sc={sc:.2f} R={R_A:.2f} M | r_hemi "
              f"{np.mean([x['r_hemi_true'] for x in rows if x['point'] == k]):+.3f} | r_assay_E "
              f"{np.mean([x['r_assay_E'] for x in rows if x['point'] == k]):+.3f} | {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


def line_boot_diff(d, rng, B=400):
    """95% line-bootstrap CI of r(M) - r(F) for one simulated assay."""
    lm = Es.line_means(d)
    wide = lm.pivot_table(index="Line", columns=["Sex", "SexRatio"], values="w")
    z = ((wide - wide.mean()) / wide.std(ddof=1)).values
    cols = list(wide.columns)
    fM, mM = z[:, cols.index(("Female", "M"))], z[:, cols.index(("Male", "M"))]
    fF, mF = z[:, cols.index(("Female", "F"))], z[:, cols.index(("Male", "F"))]
    n = len(z)
    idx = rng.integers(0, n, (B, n))

    def rowcorr(a, b):
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    diff = rowcorr(fM[idx], mM[idx]) - rowcorr(fF[idx], mF[idx])
    point = np.corrcoef(fM, mM)[0, 1] - np.corrcoef(fF, mF)[0, 1]
    return point, np.percentile(diff, [2.5, 97.5])


def part_b(cfg, args, beta, noise, pops="lh_shared", out="power_sexratio"):
    acfg = cfg["assay"]
    path = os.path.join(ROOT, "results", "pops", f"{pops}_{args.profile}.npz")
    if not os.path.exists(path):
        print("power: no saved shared populations; run scripts/run_lh.py first")
        return
    genomes, arch = load_populations(path)
    rng = np.random.default_rng(cfg["meta"]["master_seed"] + 400000)
    rows = []
    for n in POWER_LINES:
        for p in range(genomes.shape[0]):
            for a in range(args.n_power):
                ps, d = assay_r(genomes[p], arch, acfg, beta, noise, rng, n)
                point, (lo, hi) = line_boot_diff(d, rng)
                rows.append(dict(n_lines=n, pop=p, assay=a, diff_MB_FB=point, lo=lo, hi=hi,
                                 significant_positive=lo > 0, F_lowest=ps[("r_wgmf", "F")] < min(ps[("r_wgmf", "M")], ps[("r_wgmf", "E")])))
        g = pd.DataFrame(rows)
        g = g[g.n_lines == n]
        print(f"power n_lines={n}: mean diff {g.diff_MB_FB.mean():+.3f}, P(CI>0) {g.significant_positive.mean():.2f}, "
              f"P(F lowest) {g.F_lowest.mean():.2f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", f"{out}_{args.profile}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", choices=["quick", "default", "full"])
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n_assay", type=int, default=20)
    ap.add_argument("--n_power", type=int, default=25)
    ap.add_argument("--power_only", action="store_true")
    ap.add_argument("--design_only", action="store_true")
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    beta, noise = instrument(args.profile)
    print("instrument beta:", {k: round(v, 3) for k, v in beta.items()}, flush=True)
    if not args.power_only:
        part_a(cfg, args, beta, noise)
    if not args.design_only:
        part_b(cfg, args, beta, noise)
    print("done")


if __name__ == "__main__":
    main()

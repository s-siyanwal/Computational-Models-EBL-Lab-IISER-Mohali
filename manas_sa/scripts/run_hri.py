"""Layer 2, regime (ii): purely sex-limited selection + linkage (target 4).

For each map length (and each mutation rate at 0.1 M), simulate replicate
populations with the IBM and compute r_mf,W, V_f, V_m, COV (direct and
indirect), interference and LD-by-distance. Two recombination settings:
  * paper:      both sexes recombine (SLiM default used in the preprint)
  * drosophila: no crossing over in males (biology of D. melanogaster)

usage: python scripts/run_hri.py --profile quick|default|full
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

from iasc import params as P  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402
from iasc.ibm import stats as St  # noqa: E402


def run_condition(cfg, lam, R_ref_M, mu_bp, male_recomb, reps, seed0, label, ld_out, save=None, N_override=None):
    ref = cfg["reference_hri"]
    rs = A.rescale(ref["N"], ref["generations"], mu_bp, ref["L_bp"], R_ref_M, ref["gamma_scale"], lam)
    if N_override:
        rs.N = int(N_override)       # change N only: per-generation U, R, s and generations unchanged
    rng = np.random.default_rng(seed0)
    arch = A.build(rng, n_sites_A=cfg["rescale"]["sites_autosome"], mix={1: 0.5, 2: 0.5},
                   gamma_shape=ref["gamma_shape"], gamma_scale=rs.gamma_scale)
    out = np.zeros((reps, 2 * rs.N, arch.WA), np.uint64)
    seeds = np.arange(reps, dtype=np.int64) + seed0
    t0 = time.time()
    E.evolve_many(rs.N, rs.n_gen, arch.WA, 0, arch.posA, arch.posX, rs.R_A, 0.0, rs.U_A, 0.0,
                  arch.eff_f, arch.eff_m, ref["noise_sd"], male_recomb, seeds, out)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        np.savez_compressed(save, genomes=out, posA=arch.posA, posX=arch.posX, eff_f=arch.eff_f, eff_m=arch.eff_m,
                            kind=arch.kind, WA=arch.WA, WX=arch.WX, N=rs.N, lam=lam, R_A_real=R_ref_M, R_X_real=0.0)
    frac = ref["sample_f"] / (ref["N"] / 2)      # keep the paper's sampling fraction
    nf = max(10, int(round(frac * rs.N / 2)))
    rows = []
    for r in range(reps):
        st = St.population_stats(out[r], arch, rng, n_f=nf, n_m=nf)
        st.update(label=label, lam=lam, N=rs.N, map_length_M=R_ref_M, mu_bp=mu_bp,
                  male_recombination=male_recomb, rep=r, seed=int(seeds[r]))
        rows.append(st)
        if r < 3:
            ld = St.ld_by_distance(out[r], arch, R_ref_M, rng)
            if ld:
                for d in ld:
                    d.update(label=label, map_length_M=R_ref_M, mu_bp=mu_bp, male_recombination=male_recomb, rep=r)
                ld_out.extend(ld)
    dt = time.time() - t0
    rr = [x["r_mf"] for x in rows]
    print(f"[{label}] R={R_ref_M} M mu={mu_bp:g} male_rec={male_recomb}: r_mf mean {np.mean(rr):+.4f} "
          f"(se {np.std(rr, ddof=1) / np.sqrt(len(rr)):.4f}), {dt:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", choices=["quick", "default", "full"])
    ap.add_argument("--settings", default="paper,drosophila")
    args = ap.parse_args()
    cfg = P.load()
    P.below_normal_priority()
    prof = cfg["profiles"][args.profile]
    lam = cfg["rescale"]["lam_" + args.profile]
    ref = cfg["reference_hri"]
    out_csv = os.path.join(ROOT, "results", f"hri_runs_{args.profile}.csv")
    ld_csv = os.path.join(ROOT, "results", f"hri_ld_{args.profile}.csv")
    rows, ld_rows = [], []
    settings = [s for s in args.settings.split(",") if s]
    seed = cfg["meta"]["master_seed"]
    for setting in settings:
        male_recomb = setting == "paper"
        for R in prof["map_lengths_M"]:
            seed += 1000
            rows += run_condition(cfg, lam, R, ref["mu_bp"], male_recomb, prof["reps"], seed, f"maplen/{setting}", ld_rows,
                                  save=os.path.join(ROOT, "results", "pops", f"hri_{setting}_R{R:g}_{args.profile}.npz"))
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            pd.DataFrame(ld_rows).to_csv(ld_csv, index=False)
    if prof["mu_sweep"] and "paper" in settings:
        for mu in prof.get("mu_sweep_bp", ref["mu_sweep_bp"]):
            seed += 1000
            rows += run_condition(cfg, lam, ref["map_length_for_mu_sweep_M"], mu, True, prof["reps"], seed,
                                  "mu_sweep/paper", ld_rows)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()

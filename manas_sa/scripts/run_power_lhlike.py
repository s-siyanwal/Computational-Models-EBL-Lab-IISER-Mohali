"""Power for the sex-ratio ordering at an LH-like architecture (r ~ 0.4).

The 'shared' LH regime gives r ~ 0.1, at which the attenuation-only sex-ratio
gap is undetectable. Design point 8 of the sweep (f_SA = 0.061, concordant
share 0.933, female autosome map 1.48 M) gave assay r = +0.45, close to the
paper's +0.40. Evolve 6 populations with that architecture, save them, and
rerun the power analysis with the same calibrated instrument.

usage: python scripts/run_power_lhlike.py --profile default
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from iasc import params as P  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402
from iasc.ibm import hemiclone as H  # noqa: E402
from iasc.ibm import stats as St  # noqa: E402
from scripts.run_design import mix_of, part_b  # noqa: E402
from scripts.run_lh import instrument, save_populations  # noqa: E402

F_SA, SC, R_A = 0.061, 0.933, 1.48      # design point 8 (results/design_runs_default.csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", choices=["quick", "default", "full"])
    ap.add_argument("--n_pop", type=int, default=6)
    ap.add_argument("--n_power", type=int, default=25)
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    lh = cfg["lh"]
    lam = cfg["rescale"]["lam_" + args.profile]
    rs = A.rescale(lh["N"], lh["generations_ref"], lh["mu_bp"], lh["L_bp_A"], R_A, lh["gamma_scale"], lam,
                   L_bp_X=lh["L_bp_X"], R_X_ref=lh["R_X_M"])
    seed = cfg["meta"]["master_seed"] + 600000
    rng = np.random.default_rng(seed)
    arch = A.build(rng, n_sites_A=cfg["rescale"]["sites_autosome"], n_sites_X=cfg["rescale"]["sites_X"],
                   mix=mix_of(F_SA, SC), gamma_shape=lh["gamma_shape"], gamma_scale=rs.gamma_scale,
                   sa_benefit_ratio=lh["sa_benefit_ratio"])
    out = np.zeros((args.n_pop, 2 * rs.N, arch.WA + arch.WX), np.uint64)
    t0 = time.time()
    E.evolve_many(rs.N, rs.n_gen, arch.WA, arch.WX, arch.posA, arch.posX, rs.R_A, rs.R_X, rs.U_A, rs.U_X,
                  arch.eff_f, arch.eff_m, lh["noise_sd"], cfg["drosophila"]["male_recombination"],
                  np.arange(args.n_pop, dtype=np.int64) + seed, out)
    print(f"evolved {args.n_pop} LH-like populations in {time.time() - t0:.0f}s", flush=True)
    for p in range(args.n_pop):
        st = St.population_stats(out[p], arch, rng, autosome_only=False)
        bv = H.hemigenome_bv(out[p], arch, rng, n_lines=300, n_background=4000)
        r_h = np.corrcoef(bv["h_f"], bv["h_mA"] + bv["h_mX"])[0, 1]
        print(f"  pop {p}: V_f={st['V_f']:.3f} r_pop={st['r_mf']:+.3f} r_hemi={r_h:+.3f}", flush=True)
    save_populations(os.path.join(ROOT, "results", "pops", f"lh_lhlike_{args.profile}.npz"), out, arch, rs)
    beta, noise = instrument(args.profile)
    part_b(cfg, args, beta, noise, pops="lh_lhlike", out="power_sexratio_lhlike")
    print("done")


if __name__ == "__main__":
    main()

"""Diagnose r_mf,W ~ -1 at low recombination: is the population split into a
few complementary haplotype classes (an emergent SA 'supergene')?

For one replicate at a given map length and lambda, report:
  * r_mf,W (genotype-based) and its direct/indirect parts,
  * haplotype female-load and male-load (sum of per-copy effects on each
    haplotype) and their correlation ACROSS HAPLOTYPES,
  * k-means (k=2) clusters of haplotypes in (female load, male load) space,
    cluster frequencies and between-cluster share of load variance,
  * whether r stays ~ -1 when individuals are re-paired at random
    (r computed on random pairs of haplotypes = no trans-LD, only cis).

usage: python scripts/diagnose_haplotypes.py --map 0.01 --lam 4 --reps 2
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iasc import params as P  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402
from iasc.ibm import stats as St  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--male_rec", type=int, default=1)
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    ref = cfg["reference_hri"]
    rs = A.rescale(ref["N"], ref["generations"], ref["mu_bp"], ref["L_bp"], args.map, ref["gamma_scale"], args.lam)
    rng = np.random.default_rng(99)
    arch = A.build(rng, n_sites_A=cfg["rescale"]["sites_autosome"], mix={1: 0.5, 2: 0.5},
                   gamma_shape=ref["gamma_shape"], gamma_scale=rs.gamma_scale)
    out = np.zeros((args.reps, 2 * rs.N, arch.WA), np.uint64)
    E.evolve_many(rs.N, rs.n_gen, arch.WA, 0, arch.posA, arch.posX, rs.R_A, 0.0, rs.U_A, 0.0,
                  arch.eff_f, arch.eff_m, ref["noise_sd"], bool(args.male_rec),
                  np.arange(args.reps, dtype=np.int64) + 4242, out)
    rows = []
    for r in range(args.reps):
        st = St.population_stats(out[r], arch, rng)
        bits = E.unpack(out[r], np.arange(2 * rs.N), arch.WA).astype(np.float32)
        hf = bits @ arch.eff_f[: arch.WA * 64]
        hm = bits @ arch.eff_m[: arch.WA * 64]
        r_hap = np.corrcoef(hf, hm)[0, 1]
        # 2-means in standardised load space
        Z = np.column_stack([(hf - hf.mean()) / hf.std(), (hm - hm.mean()) / hm.std()])
        c = Z[rng.choice(len(Z), 2, replace=False)]
        for _ in range(50):
            lab = np.argmin(((Z[:, None, :] - c[None]) ** 2).sum(-1), 1)
            c = np.array([Z[lab == k].mean(0) if np.any(lab == k) else c[k] for k in range(2)])
        between = sum((lab == k).mean() * ((Z[lab == k].mean(0)) ** 2).sum() for k in range(2)) / 2
        # random re-pairing of haplotypes into diploids (removes trans association)
        perm = rng.permutation(2 * rs.N)
        g_f = hf[perm[0::2]] + hf[perm[1::2]]
        g_m = hm[perm[0::2]] + hm[perm[1::2]]
        n_unique = len({bits[i].tobytes() for i in range(bits.shape[0])})
        rows.append(dict(rep=r, map_M=args.map, lam=args.lam, r_mf=st["r_mf"], direct=st["direct"], indirect=st["indirect"],
                         r_across_haplotypes=r_hap, r_random_repairing=np.corrcoef(g_f, g_m)[0, 1],
                         cluster_freqs=np.round(np.bincount(lab, minlength=2) / len(lab), 3).tolist(),
                         cluster_centres=np.round(c, 2).tolist(), between_cluster_share=round(float(between), 3),
                         distinct_haplotypes=n_unique, n_seg=st["n_seg"]))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(ROOT, "results", f"diag_haplotypes_map{args.map}_lam{args.lam}.csv"), index=False)


if __name__ == "__main__":
    main()

"""Cross-check the numba engine against SLiM 5.2 (when available).

Runs slim/hri_sexlimited.slim and the numba engine at the same (small,
unrescaled) parameters for several seeds, computes r_mf,W from both with the
same estimator, and reports a two-sample KS test. If `slim` is not on PATH
the check is skipped (this was the case on the development machine).

usage: python scripts/slim_crosscheck.py --N 300 --gens 3000 --maplen 0.1 --reps 10
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


def r_from_slim_output(path):
    muts, inds = {}, []
    section = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#mutations"):
                section = "m"
                continue
            if line.startswith("#individuals"):
                section = "i"
                continue
            if not line:
                continue
            if section == "m":
                mid, mtype, pos, s = line.split()
                muts[int(mid)] = (mtype, float(s))
            elif section == "i":
                sex, rest = line.split(" ", 1)
                h1, h2 = rest.split(";")
                ids = [int(x) for x in (h1.replace(",", " ").split() + h2.replace(",", " ").split())]
                inds.append(ids)
    bvf = np.array([sum(muts[i][1] for i in ids if muts[i][0] in ("1", "m1")) for ids in inds])
    bvm = np.array([sum(muts[i][1] for i in ids if muts[i][0] in ("2", "m2")) for ids in inds])
    return np.corrcoef(bvf, bvm)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--gens", type=int, default=3000)
    ap.add_argument("--maplen", type=float, default=0.1)
    ap.add_argument("--mu", type=float, default=7e-7)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    slim = shutil.which("slim")
    if slim is None:
        print("SKIPPED: `slim` not found on PATH. Install SLiM 5.2 (MSYS2 or WSL on Windows) and rerun.")
        return 0
    from iasc.ibm import architecture as A, engine as E, stats as St
    r_slim, r_ibm = [], []
    with tempfile.TemporaryDirectory() as td:
        for s in range(args.reps):
            out = os.path.join(td, f"r{s}.txt").replace("\\", "/")
            subprocess.run([slim, "-d", f"N={args.N}", "-d", f"GENS={args.gens}", "-d", f"MU={args.mu}",
                            "-d", f"MAPLEN={args.maplen}", "-d", "MALE_REC=1", "-d", f"SEED={s + 1}",
                            "-d", f"OUT='{out}'", os.path.join(ROOT, "slim", "hri_sexlimited.slim")], check=True,
                           capture_output=True)
            r_slim.append(r_from_slim_output(out))
    rng = np.random.default_rng(1)
    arch = A.build(rng, n_sites_A=32768, mix={1: 0.5, 2: 0.5})
    out = np.zeros((args.reps, 2 * args.N, arch.WA), np.uint64)
    E.evolve_many(args.N, args.gens, arch.WA, 0, arch.posA, arch.posX, args.maplen, 0.0, args.mu * 1e6, 0.0,
                  arch.eff_f, arch.eff_m, 1.0, True, np.arange(args.reps, dtype=np.int64) + 1, out)
    for r in range(args.reps):
        r_ibm.append(St.population_stats(out[r], arch, rng, n_f=args.N // 10, n_m=args.N // 10)["r_mf"])
    ks = stats.ks_2samp(r_slim, r_ibm)
    print(f"SLiM mean r {np.mean(r_slim):+.3f}  numba mean r {np.mean(r_ibm):+.3f}  KS D={ks.statistic:.2f} p={ks.pvalue:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

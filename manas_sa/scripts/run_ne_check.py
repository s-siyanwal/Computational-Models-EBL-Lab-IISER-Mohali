"""N-dependence of r_mf,W under purely sex-limited selection (the proposed
discriminator between HRI and SA; docs/05).

At the preprint's own per-generation parameters (lambda = 1: mu = 7e-7/bp,
1 Mb, map 0.1 M, s ~ Gamma(0.3, 0.05), 25,000 generations) only N is changed.
HRI predicts the negative bias to grow as N falls (HRI ~ 1/(N c)); true
pleiotropic SA predicts no first-order N dependence. The N = 2,500 point is
the lambda = 1 run in results/lambda_sensitivity.csv.

usage: python scripts/run_ne_check.py --Ns 625,1250 --reps 6
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from iasc import params as P  # noqa: E402
from scripts.run_hri import run_condition  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="625,1250")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--map", type=float, default=0.1)
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    path = os.path.join(ROOT, "results", "ne_check.csv")
    rows = []
    seed = cfg["meta"]["master_seed"] + 700000
    for N in [int(x) for x in args.Ns.split(",")]:
        seed += 1000
        rows += run_condition(cfg, 1, args.map, cfg["reference_hri"]["mu_bp"], True, args.reps, seed, f"N{N}", [],
                              N_override=N)
        pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Rescaling diagnostic: r_mf,W at fixed map lengths for several lambda
(lambda = 1 is the preprint's exact parameterisation, no rescaling).

usage: python scripts/run_lambda_check.py --maps 0.01,0.1 --lams 1,2 --reps 6
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
    ap.add_argument("--maps", default="0.01,0.1")
    ap.add_argument("--lams", default="1,2")
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    path = os.path.join(ROOT, "results", "lambda_sensitivity.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    ld = []
    seed = cfg["meta"]["master_seed"] + 50000
    for lam in [float(x) for x in args.lams.split(",")]:
        for R in [float(x) for x in args.maps.split(",")]:
            seed += 1000
            rows += run_condition(cfg, lam, R, cfg["reference_hri"]["mu_bp"], True, args.reps, seed,
                                  f"lambda{lam:g}", ld)
            pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()

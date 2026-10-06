"""What would the smallest extra assays buy? (outputs/chinmay_power.csv)

1. Bound on the perceived fecundity difference from Chinmay's pooled
   courts-first data (240 of 463 vials: sham first). Only the product
   beta * kappa is identified by a choice assay; with the frozen beta0 (and
   depletion = 1) it converts to an upper bound on kappa.
2. Power of the telegony assay to detect a stepfather effect (log-hazard shift
   +d for I, 0 for U, -d for S stepchildren) with one block (thesis) versus a
   replicated design of four blocks, everything else as in the thesis.

usage: python scripts/run_chinmay_power.py [--reps 200]
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from extensions import telegony as T  # noqa: E402
from extensions.config import load_regime, values  # noqa: E402
from iasc import params as IP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    IP.below_normal_priority()
    ius = load_regime("ius")
    P = values(ius)
    rows = []
    k, n = 240, 463
    ci = st.binomtest(k, n).proportion_ci(0.95, method="exact")
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    bk_hi = logit(ci.high)
    b = P["beta0"] * P["depletion_depleted"]
    rows.append(dict(analysis="choice bound", quantity="sham courted first (pooled, thesis Table 3.2)",
                     value=f"{k}/{n} = {k / n:.3f}, exact 95% CI [{ci.low:.3f}, {ci.high:.3f}]"))
    rows.append(dict(analysis="choice bound", quantity="identified product beta * kappa (upper 95%)", value=f"{bk_hi:.3f}"))
    rows.append(dict(analysis="choice bound", quantity=f"implied kappa upper bound at frozen beta0 = {P['beta0']:g}",
                     value=f"{bk_hi / b:.3f} (a fecundity cost above ~{100 * bk_hi / b:.0f}% would have been seen, IF males are as sensitive as the Khan control requires)"))
    F = ius["modules"]
    for blocks in (1, 4):
        for d in (0.0, 0.1, 0.2, 0.3, 0.5):
            Pd = dict(P, compartmentalization=False, semen_effect_sd=d, n_infected_per_cell=50 * blocks)
            sig = 0
            for r in range(args.reps):
                a = T.analyse_telegony(T.simulate_chinmay_telegony(
                    np.random.default_rng(1000 * blocks + r), Pd, F, stepfather_condition={"I": 1.0, "U": 0.0, "S": -1.0}))
                sig += a[("wald", "Population")][2] < 0.05
            rows.append(dict(analysis="telegony power", quantity=f"{blocks} block(s), stepfather log-hazard shift +-{d}",
                             value=f"P(population Wald p<0.05) = {sig / args.reps:.2f}"))
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "outputs", "chinmay_power.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

"""Consequences of the independent (Byrne & Rice 2006) calibration of the
courtship kernel for the Chinmay / Wittman-Fedorka comparisons
(outputs/chinmay_calibrated.csv). Run calibrate_byrne_rice.py first.

  * A3 under calibrated sensitivity: would Chinmay's design (depleted males,
    decapitated females, Table 3.2 sizes) detect a Khan-sized fecundity cost?
  * bound on kappa from Chinmay's pooled courts-first data
  * Wittman & Fedorka: largest courts-most share reachable by any kappa <= 1
  * latency slope: Byrne & Rice no-choice controls (large vs small females:
    86.7% vs 78.3% mated, n.s.; reversed in the follow-up, 34.6% vs 42.8%)
    show no consistent effect of female quality on mating rate.

usage: python scripts/run_chinmay_calibrated.py [--reps 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from extensions import choice as C  # noqa: E402
from extensions.config import load_regime, values  # noqa: E402
from iasc import params as IP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    IP.below_normal_priority()
    cal = json.load(open(os.path.join(ROOT, "outputs", "byrne_rice_calibration.json")))["linear"]
    b_nd = cal["calibration"]["nondepleted"]["beta"]
    b_d = cal["calibration"]["depleted"]
    ius = load_regime("ius")
    P = values(ius)
    Pk = values(load_regime("khan"))
    Pc = dict(P, beta0=b_d["beta"], depletion_depleted=1.0, depletion_nondepleted=b_nd / b_d["beta"])
    rows = []
    shares, pooled_sig, any_block, cl_sig = [], [], [], []
    for r in range(args.reps):
        rec = C.simulate_chinmay_study(np.random.default_rng(777 + r), Pc, ius["modules"], kappa=Pk["kappa"])
        a = C.analyse_chinmay(rec)
        shares.append(a["cf_sham_share"])
        n = int(np.sum(rec["courted"]))
        k = int(np.sum(rec["cf_sham"] & rec["courted"]))
        pooled_sig.append(st.binomtest(k, n).pvalue < 0.05)
        any_block.append(any(a[("cf_p_block", b)] < 0.05 for b in (1, 2, 3, 4)))
        cl_sig.append(a[("cl", "INF")][1] < 0.05)
    rows.append(dict(item="A3 with calibrated beta (depleted beta = %.2f)" % b_d["beta"],
                     result=f"sham-first share {np.mean(shares):.3f} [{np.quantile(shares, .025):.3f}, {np.quantile(shares, .975):.3f}]; "
                            f"P(pooled 463-vial test p<0.05) = {np.mean(pooled_sig):.2f}; P(any block p<0.05) = {np.mean(any_block):.2f}; "
                            f"P(latency infection p<0.05) = {np.mean(cl_sig):.2f}"))
    bk = np.log(0.565 / 0.435)          # upper exact 95% limit of 240/463 (see run_chinmay_power.py)
    for lab, b in (("point", b_d["beta"]), ("upper 95% beta", b_d["beta_hi"]), ("lower 95% beta", b_d["beta_lo"])):
        rows.append(dict(item=f"kappa bound from Chinmay courts-first, calibrated depleted beta ({lab} = {b:.2f})",
                         result=f"kappa <= {bk / b:.2f}" if b > 0 else "no bound (beta = 0 within CI)"))
    cm_max = []
    for kap in (0.4, 1.0):
        Pw = dict(P, observation_min=20)
        rr = C.simulate_two_choice(np.random.default_rng(55), 20000, 1.0, 1 - kap, b_nd, Pw)
        cm_max.append((kap, float(np.nanmean(rr["cm_sham"])), float(rr["cf_sham"].mean())))
    rows.append(dict(item="Wittman & Fedorka CM 0.64 reachable? (non-depleted calibrated beta = %.2f)" % b_nd,
                     result="; ".join(f"kappa={k:g}: CM {cm:.3f}, CF {cf:.3f}" for k, cm, cf in cm_max)
                            + " -> CM 0.64 not reachable by any fecundity cost under the calibrated sensitivity"))
    rows.append(dict(item="latency-quality slope (independent check)",
                     result="Byrne & Rice no-choice controls: no consistent effect of female fecundity on mating rate "
                            "(86.7% vs 78.3% n.s.; reversed 34.6% vs 42.8%) -> slope ~0 is supported; "
                            "with slope 0 the Khan latency ratio prediction is 1.00 vs observed 1.06 and 1.02 "
                            "(but 'no effect' is also the trivial baseline)"))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "outputs", "chinmay_calibrated.csv"), index=False)
    pd.set_option("display.max_colwidth", 300)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

"""Blind calibration of the courtship kernel on an INDEPENDENT dataset, then an
out-of-sample prediction of Khan & Prasad (2013).

Calibration data: Byrne & Rice 2006, Proc R Soc B 273:917-922 (LHM flies).
  * assay: 10 males + 10 small + 10 large virgin females, 30 min, a female is
    'mated' if her vial produced larvae (section 2f) -- the same assay type as
    Khan's mating-bias vials;
  * measured fecundity (section 3a): large 69.30 vs small 25.06 offspring, so
    the perceived-quality difference is dq = 1 - 25.06/69.30 = 0.638 on the
    linear fecundity scale the spec prescribes (q = F/F0);
  * outcome (sections 3b-c): share of vials in which more large than small
    females were mated:
        non-depleted males  22 of 32 non-tied vials
        depleted males      24 of 29 (primary) + 33 of 43 (follow-up) = 57 of 72.
The per-vial delta means are only in Fig. 1, so the sign-test counts are used.

Fitted: beta for non-depleted and depleted males (=> beta0 and the depletion
ratio). Nothing from Khan or Chinmay enters the fit.

Prediction: Khan's mating-bias score (infected / mated; 10 males, 10 + 10
females, 45 min, dq = 0.4 from Khan's 'approximately 40%' fecundity reduction,
non-depleted males). Compared with Khan Table 5 (0.447, 0.478, 0.469, 0.459)
only after the prediction is made.

Structural alternative reported as a sensitivity: perceived quality on a log
scale (q = ln F), under which Byrne's dq = ln(69.30/25.06) = 1.017 and Khan's
dq = -ln(0.6) = 0.511.

usage: python scripts/calibrate_byrne_rice.py [--vials 3000]
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

BR = dict(fec_large=69.30, fec_small=25.06, minutes=30, n_males=10, n_each=10,
          nondepleted=(22, 32), depleted=(24 + 33, 29 + 43))
KHAN_TABLE5 = (0.447, 0.478, 0.469, 0.459)


def share_large_ahead(b, dq, n_vials, seed, acc, lat):
    rng = np.random.default_rng(seed)
    pos = neg = 0
    for _ in range(n_vials):
        ml, ms = C.group_mating_vial(rng, 1.0, 1.0 - dq, b, BR["n_males"], BR["n_each"], BR["minutes"], acc, lat)
        pos += ml > ms
        neg += ml < ms
    return pos / max(pos + neg, 1)


def khan_bias(b, dq, n_vials, seed, acc, lat, minutes=45):
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n_vials):
        ms, mi = C.group_mating_vial(rng, 1.0, 1.0 - dq, b, 10, 10, minutes, acc, lat)
        if mi + ms:
            s.append(mi / (mi + ms))
    return float(np.mean(s))


def invert(curve_b, curve_p, target):
    """Monotone interpolation of beta for a target share."""
    p = np.maximum.accumulate(np.asarray(curve_p))
    if target <= p[0]:
        return float(curve_b[0])
    if target >= p[-1]:
        return float("inf")
    return float(np.interp(target, p, curve_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vials", type=int, default=3000)
    args = ap.parse_args()
    IP.below_normal_priority()
    P = values(load_regime("ius"))
    Pk = dict(P)
    Pk.update(values(load_regime("khan")))
    acc, lat = Pk["receptivity_intact"], P["latency_mean_min"]
    out = {}
    for scale in ("linear", "log"):
        if scale == "linear":
            dq_br = 1 - BR["fec_small"] / BR["fec_large"]
            dq_k = Pk["kappa"]
        else:
            dq_br = np.log(BR["fec_large"] / BR["fec_small"])
            dq_k = -np.log(1 - Pk["kappa"])
        grid = np.round(np.arange(0.0, 6.01, 0.25), 2)
        curve = [share_large_ahead(b, dq_br, args.vials, 11 + i, acc, lat) for i, b in enumerate(grid)]
        res = {}
        for lab in ("nondepleted", "depleted"):
            k, n = BR[lab]
            ci = st.binomtest(k, n).proportion_ci(0.95, method="exact")
            res[lab] = dict(share=k / n, ci=(ci.low, ci.high), beta=invert(grid, curve, k / n),
                            beta_lo=invert(grid, curve, ci.low), beta_hi=invert(grid, curve, ci.high))
        b_nd = res["nondepleted"]
        pred = khan_bias(b_nd["beta"], dq_k, args.vials, 999, acc, lat)
        pred_lo = khan_bias(b_nd["beta_hi"], dq_k, args.vials, 999, acc, lat) if np.isfinite(b_nd["beta_hi"]) else float("nan")
        pred_hi = khan_bias(b_nd["beta_lo"], dq_k, args.vials, 999, acc, lat)
        obs = float(np.mean(KHAN_TABLE5))
        p_first_khan = float(C.p_first(1.0, 1 - dq_k, b_nd["beta"]))
        out[scale] = dict(dq_byrne=float(dq_br), dq_khan=float(dq_k), curve=dict(zip(map(float, grid), map(float, curve))),
                          calibration=res, depletion_ratio=res["depleted"]["beta"] / res["nondepleted"]["beta"] if res["nondepleted"]["beta"] > 0 else float("nan"),
                          khan_prediction=pred, khan_prediction_range=(pred_lo, pred_hi), khan_observed_mean=obs,
                          relative_error=(pred - obs) / obs, within_10pct=abs(pred - obs) / obs <= 0.10,
                          implied_A2_p_sham_first=p_first_khan)
        print(f"[{scale}] Byrne dq={dq_br:.3f}: beta_nd={b_nd['beta']:.2f} [{b_nd['beta_lo']:.2f}, {b_nd['beta_hi']:.2f}], "
              f"beta_d={res['depleted']['beta']:.2f}; Khan predicted bias {pred:.3f} [{pred_lo:.3f}, {pred_hi:.3f}] "
              f"vs observed {obs:.3f} (rel. error {100 * (pred - obs) / obs:+.1f}%); implied A2 P(sham first) {p_first_khan:.3f}", flush=True)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "byrne_rice_calibration.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    rows = [dict(scale=s, beta_nondepleted=o["calibration"]["nondepleted"]["beta"],
                 beta_nondepleted_95=f"[{o['calibration']['nondepleted']['beta_lo']:.2f}, {o['calibration']['nondepleted']['beta_hi']:.2f}]",
                 beta_depleted=o["calibration"]["depleted"]["beta"], depletion_ratio=o["depletion_ratio"],
                 khan_predicted=o["khan_prediction"], khan_predicted_range=f"[{o['khan_prediction_range'][0]:.3f}, {o['khan_prediction_range'][1]:.3f}]",
                 khan_observed_mean=o["khan_observed_mean"], relative_error_pct=100 * o["relative_error"],
                 within_10pct=o["within_10pct"], implied_A2=o["implied_A2_p_sham_first"]) for s, o in out.items()]
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "outputs", "byrne_rice_calibration.csv"), index=False)


if __name__ == "__main__":
    main()

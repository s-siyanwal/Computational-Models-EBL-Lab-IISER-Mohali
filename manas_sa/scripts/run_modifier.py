"""Modifier layer: exact re-run of the authors' parameter-space analysis
(Singh, Jain, Geeta Arun & Prasad 2023, Figures 1-2) for autosomes and X,
plus automatic scoring of each qualitative claim.

usage: python scripts/run_modifier.py [--step 0.1]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from iasc import params as P  # noqa: E402
from iasc.analytic import modifier_cc as mc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()
    P.below_normal_priority()
    ks = np.round(np.arange(0, 1.0001, args.step), 2)
    kp = np.array([[a, b] for a in ks for b in ks])
    rows = []
    res = {}
    for xl in (False, True):
        t = time.time()
        cases = mc.build_cases(xl)
        frac, tm = mc.run_grid(kp, cases, xl)
        res[xl] = (frac, tm)
        for (k1, k2), f, tt in zip(kp, frac, tm):
            for q, name in enumerate(mc.OUTCOMES):
                rows.append(dict(chromosome="X" if xl else "autosome", k1=k1, k2=k2, outcome=name,
                                 frac_fixed=f[q], mean_time=tt[q], n_cases=len(cases)))
        print(f"{'X' if xl else 'autosome'}: {len(cases)} cases x {len(kp)} (k1,k2) in {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "modifier_grid.csv"), index=False)

    # ---- claim scoring
    def grid(xl, q=2, what=0):
        return res[xl][what][:, q].reshape(len(ks), len(ks))   # [k1, k2]
    A, X = grid(False), grid(True)
    i0, i1 = 0, len(ks) - 1
    claims = []
    claims.append(dict(claim="k1=0,k2=1: A2B2 fixed in 95.7% (autosomes)", reported=0.957,
                       model=float(A[i0, i1]), verdict="matches magnitude" if abs(A[i0, i1] - 0.957) < 0.01 else
                       f"misses magnitude by {A[i0, i1] - 0.957:+.3f}"))
    claims.append(dict(claim="k2=1: A2B2 fixed in >85% for any k1 (autosomes)", reported=">0.85",
                       model=float(A[:, i1].min()), verdict="matches" if A[:, i1].min() > 0.85 else "does not match"))
    mono_k2 = np.mean([np.all(np.diff(A[i, :]) >= -1e-9) for i in range(len(ks))])
    claims.append(dict(claim="resolution increases with k2", reported="monotone increase",
                       model=f"monotone in {mono_k2:.0%} of k1 rows (autosome)",
                       verdict="matches ordering" if mono_k2 > 0.8 else "partly matches ordering"))
    colmean = A.mean(1)
    inner_min = int(np.argmin(colmean))
    claims.append(dict(claim="k1 non-monotonic: hardest at intermediate k1", reported="minimum at intermediate k1",
                       model=f"row-mean minimum at k1={ks[inner_min]}",
                       verdict="matches ordering" if 0 < inner_min < len(ks) - 1 else "does not match"))
    claims.append(dict(claim="X more conducive than autosomes overall", reported="X > A",
                       model=f"mean frac X {X.mean():.3f} vs A {A.mean():.3f}",
                       verdict="matches ordering" if X.mean() > A.mean() else "does not match"))
    lowk2 = slice(0, len(ks) // 2)
    claims.append(dict(claim="X advantage strongest at low k2", reported="X - A largest at low k2",
                       model=f"mean(X-A) low k2 {np.mean(X[:, lowk2] - A[:, lowk2]):.3f}, high k2 {np.mean(X[:, len(ks)//2:] - A[:, len(ks)//2:]):.3f}",
                       verdict="matches ordering" if np.mean(X[:, lowk2] - A[:, lowk2]) > np.mean(X[:, len(ks)//2:] - A[:, len(ks)//2:]) else "does not match"))
    claims.append(dict(claim="at very high k2 autosomes resolve a larger fraction than X", reported="A > X at k2 ~ 1",
                       model=f"mean(A-X) at k2=1: {np.mean(A[:, i1] - X[:, i1]):+.3f}",
                       verdict="matches ordering" if np.mean(A[:, i1] - X[:, i1]) > 0 else "does not match"))
    pd.DataFrame(claims).to_csv(os.path.join(ROOT, "results", "modifier_claims.csv"), index=False)
    print(pd.DataFrame(claims).to_string(index=False))

    # ---- figure analogous to the paper's Figure 1 (fractions) and 2 (times)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for col, (q, name) in enumerate(((0, "A2"), (1, "B2"), (2, "A2B2"))):
        for row, xl in enumerate((False, True)):
            ax = axs[row, col]
            im = ax.imshow(grid(xl, q), origin="lower", vmin=0, vmax=1, cmap="Blues",
                           extent=(-0.05, 1.05, -0.05, 1.05), aspect="auto")
            ax.set(title=f"{'X-linked' if xl else 'autosomal'}: {name} fixed (females)", xlabel="k2", ylabel="k1")
    fig.colorbar(im, ax=axs, label="fraction of parameter space fixed within 3000 generations")
    fig.savefig(os.path.join(ROOT, "results", "fig_modifier_fraction.png"), dpi=140, bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    main()

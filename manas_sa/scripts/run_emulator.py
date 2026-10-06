"""Layer 3: thin GP emulator of the IBM design sweep (scripts/run_design.py).

Used ONLY for design / sensitivity / inversion questions, never as evidence
about the papers on its own:
  1. grouped cross-validation (whole design points held out): GP vs GBM RMSE
     against the replicate-noise floor;
  2. first-order sensitivity (variance of the conditional GP mean, binned
     Monte Carlo) of r_hemi_true and of the assay r_w,g,mf (equal sex ratio);
  3. inversion: which fraction of SA mutations (f_SA) makes the 39-line assay
     return each published r (target 1 and the contrast set), at the other
     architecture parameters of the 'shared' regime;
  4. figure: emulated r vs f_SA with the published values overlaid.

usage: python scripts/run_emulator.py --profile default
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from iasc import params as P  # noqa: E402
from iasc.emulator import surrogate as S  # noqa: E402

FEATURES = ["f_SA", "sc_share", "log10_R"]
TARGETS = ["r_hemi_true", "r_assay_E"]
# published r values the inversion is asked about (claim type A)
N_GENOMES = {"Geeta Arun 2022 (LH, equal SR, 39)": 39, "Chippindale 2001 (LHM adult, 40)": 40,
             "Innocenti & Morrow 2010 (LHM, 100)": 100, "Collet 2016 LHM-UU (100)": 100,
             "Collet 2016 LHM-UCL (113)": 113, "Ruzicka 2019 (LHM, 223)": 223}
PUBLISHED = {"Geeta Arun 2022 (LH, equal SR, 39)": 0.4027, "Chippindale 2001 (LHM adult, 40)": -0.30,
             "Innocenti & Morrow 2010 (LHM, 100)": -0.52, "Collet 2016 LHM-UU (100)": -0.41,
             "Collet 2016 LHM-UCL (113)": 0.21, "Ruzicka 2019 (LHM, 223)": 0.15}
INK, MUTED = "#0b0b0b", "#898781"
COLS = ["#2a78d6", "#eb6834", "#1baf7a"]


def first_order(model, rng, n=4000, bins=20):
    X = np.column_stack([rng.uniform(0, 0.2, n), rng.uniform(0, 1, n), rng.uniform(-1, np.log10(2), n)])
    y = model.predict(X)
    out = {}
    for j, f in enumerate(FEATURES):
        q = np.quantile(X[:, j], np.linspace(0, 1, bins + 1))
        b = np.clip(np.searchsorted(q, X[:, j]) - 1, 0, bins - 1)
        means = np.array([y[b == k].mean() for k in range(bins)])
        out[f] = float(means.var() / y.var())
    return out


def save_png(fig, path):
    """savefig via temp file + os.replace with retries (Windows file locks, Errno 22)."""
    import time
    for attempt in range(6):
        try:
            tmp = path + f".tmp{attempt}.png"
            fig.savefig(tmp)
            os.replace(tmp, path)
            break
        except OSError:
            time.sleep(1.0 + attempt)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", choices=["quick", "default", "full"])
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    res = os.path.join(ROOT, "results")
    df = pd.read_csv(os.path.join(res, f"design_runs_{args.profile}.csv"))
    mix = cfg["lh"]["mix_shared"]
    f_sa0 = mix.get(4, 0) + mix.get(5, 0)
    sc0 = mix.get(3, 0) / (1 - f_sa0)
    lr0 = np.log10(cfg["lh"]["R_A_M"])
    rows, models = [], {}
    rng = np.random.default_rng(cfg["meta"]["master_seed"])
    for t in TARGETS:
        model, cv = S.fit_and_validate(df, FEATURES, t, "point")
        models[t] = model
        so = first_order(model, rng)
        rows.append(dict(target=t, GP_cv_rmse=cv["GP"], GBM_cv_rmse=cv["GBM"], replicate_sd=cv["replicate_sd"],
                         **{f"S1_{k}": v for k, v in so.items()}))
    cvt = pd.DataFrame(rows)
    cvt.to_csv(os.path.join(res, f"emulator_validation_{args.profile}.csv"), index=False)
    print(cvt.round(3).to_string(index=False))

    # inversion along f_SA at the shared-regime sc_share and map length
    grid = np.linspace(0, 0.2, 201)
    Xg = np.column_stack([grid, np.full_like(grid, sc0), np.full_like(grid, lr0)])
    gp = models["r_assay_E"]
    mu, sd = gp.predict(Xg, return_std=True)
    # predictive band adds the 39-line between-assay sd (replicate assays of one population)
    samp = float(df["r_assay_E_sd"].mean())
    inv = []
    for name, val in PUBLISHED.items():
        ok = np.abs(mu - val) <= 1.96 * np.sqrt(sd ** 2 + samp ** 2)
        best = grid[np.argmin(np.abs(mu - val))]
        inv.append(dict(study=name, published_r=val, f_SA_best=float(best),
                        f_SA_consistent_min=float(grid[ok].min()) if ok.any() else np.nan,
                        f_SA_consistent_max=float(grid[ok].max()) if ok.any() else np.nan,
                        emulator_r_at_shared_regime=float(gp.predict([[f_sa0, sc0, lr0]])[0]),
                        note="consistent = emulator mean within 1.96*sqrt(GP var + 39-line assay var)"))
    # sampling alone: probability that an assay of n genomes from a population
    # with the shared-regime architecture returns a value at least as far from
    # the emulator mean as the published one (normal approximation; sd of the
    # n-genome assay taken from the IBM replicate assays)
    m0, s0 = gp.predict([[f_sa0, sc0, lr0]], return_std=True)
    mu0, sd_pop = float(m0[0]), float(s0[0])      # s0 includes between-population (white-noise) variance
    near = df[np.abs(df.r_assay_E - mu0) < 0.25]
    near = near if len(near) >= 3 else df
    for row in inv:
        n = N_GENOMES[row["study"]]
        col = f"r_E_n{n}_sd" if f"r_E_n{n}_sd" in df else "r_assay_E_sd"
        sd_n = float(near[col].mean())
        z = (row["published_r"] - mu0) / np.sqrt(sd_n ** 2 + sd_pop ** 2)
        row.update(assay_sd_at_n=sd_n, between_population_sd=sd_pop, z_vs_shared_regime=z,
                   P_as_extreme_by_sampling=float(2 * norm.sf(abs(z))))
    invt = pd.DataFrame(inv)
    invt.to_csv(os.path.join(res, f"emulator_inversion_{args.profile}.csv"), index=False)
    print(invt.round(3).to_string(index=False))

    # figure
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                         "savefig.dpi": 150, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for sc, col in zip((0.2, sc0, 0.95), COLS):
        Xs = np.column_stack([grid, np.full_like(grid, sc), np.full_like(grid, lr0)])
        m, s = gp.predict(Xs, return_std=True)
        ax.plot(grid, m, color=col, lw=2, label=f"emulator, concordant share {sc:.2f}")
        ax.fill_between(grid, m - 1.96 * s, m + 1.96 * s, color=col, alpha=0.15, lw=0)
    ax.scatter(df.f_SA, df.r_assay_E, s=14, color=MUTED, label="IBM runs (all map lengths / shares)", zorder=3)
    for k, (name, val) in enumerate(PUBLISHED.items()):
        ax.axhline(val, color=INK if k == 0 else MUTED, lw=1, ls="-" if k == 0 else ":")
        ax.text(0.202, val, name, fontsize=6.5, va="center", color=INK)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set(xlabel="fraction of new mutations that are sexually antagonistic (f_SA)",
           ylabel="r_w,g,mf from a 39-line hemiclone assay", xlim=(0, 0.2),
           title="Which architecture produces each published r? (GP emulator of the IBM)")
    ax.legend(fontsize=7, loc="lower left")
    save_png(fig, os.path.join(res, f"fig_emulator_inversion_{args.profile}.png"))


if __name__ == "__main__":
    main()

"""The smallest experiment that distinguishes true sexual antagonism (SA) from
Hill-Robertson / linkage-driven negative r (HRI), run in silico on the saved
populations of every regime:

  r(k) = corr(female value, male value) of hemigenomes after k generations of
         female meiosis among the sampled hemigenomes (no selection);
  r(LE) = same after permuting every site independently (k -> infinity).

SA predicts r(k) ~ r(0) (pleiotropy is not broken by recombination);
HRI predicts r(k) -> r(LE) ~ 0 (only LD is broken).
Power: with the calibrated assay instrument (equal sex ratio), the
probability that a study with n original and n recombinant lines detects
the change r(0) -> r(k).

usage: python scripts/run_distinguish.py --profile default
"""
from __future__ import annotations

import argparse
import glob
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

from iasc import estimators as Es  # noqa: E402
from iasc import params as P  # noqa: E402
from iasc.ibm import experiments as X  # noqa: E402
from iasc.ibm import hemiclone as H  # noqa: E402
from scripts.run_lh import instrument, load_populations  # noqa: E402

KS = (0, 1, 2, 4, 8, 16)
K_EXP = 4
N_LINES = (39, 100, 200)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def real_map(path, cfg):
    z = np.load(path)
    if "R_A_real" in z:
        return float(z["R_A_real"]), float(z["R_X_real"])
    return cfg["lh"]["R_A_M"], cfg["lh"]["R_X_M"]


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
    ap.add_argument("--pool", type=int, default=600)
    ap.add_argument("--n_assay", type=int, default=20)
    args = ap.parse_args()
    P.below_normal_priority()
    cfg = P.load()
    beta, noise = instrument(args.profile)
    acfg = cfg["assay"]
    rng = np.random.default_rng(cfg["meta"]["master_seed"] + 500000)
    rows, prow = [], []
    files = sorted(glob.glob(os.path.join(ROOT, "results", "pops", f"*_{args.profile}.npz")))
    for path in files:
        name = os.path.basename(path).replace(f"_{args.profile}.npz", "")
        genomes, arch = load_populations(path)
        R_A, R_X = real_map(path, cfg)
        for rep in range(genomes.shape[0]):
            G = genomes[rep]
            Hs = H.sample_hemigenomes(G, arch, rng, args.pool)
            vf0, vm0, bits = X.haplotype_values(Hs, arch)
            W = arch.WA + arch.WX
            ef, em = arch.eff_f[: W * 64].astype(np.float32), arch.eff_m[: W * 64].astype(np.float32)
            r_le = float(np.mean([corr(le @ ef, le @ em) for le in (X.linkage_equilibrium(bits, rng) for _ in range(5))]))
            for k in KS:
                Hk = X.recombine_pool(Hs, arch, k, R_A, R_X, seed=int(rng.integers(1 << 30)))
                vf, vm, _ = X.haplotype_values(Hk, arch)
                rows.append(dict(population=name, rep=rep, k=k, r=corr(vf, vm), r_LE=r_le, r0=corr(vf0, vm0),
                                 R_A_per_meiosis=R_A))
            # power of the n-line experiment (original vs k = K_EXP recombinant lines)
            Hk = X.recombine_pool(Hs, arch, K_EXP, R_A, R_X, seed=int(rng.integers(1 << 30)))
            for n in N_LINES:
                r_o, r_k = [], []
                for _ in range(args.n_assay):
                    for src, acc in ((Hs, r_o), (Hk, r_k)):
                        sel = src[rng.choice(len(src), n, replace=False)]
                        bv = H.assay_bv(G, arch, rng, sel, 4000)
                        d = H.run_assay(bv, acfg, beta, noise, rng, srs=("E",))
                        acc.append(Es.paper_stats(Es.line_means(d))[("r_wgmf", "E")])
                z = (np.mean(r_o) - np.mean(r_k)) / np.sqrt(np.var(r_o, ddof=1) + np.var(r_k, ddof=1))
                prow.append(dict(population=name, rep=rep, n_lines=n, k=K_EXP, r_assay_original=np.mean(r_o),
                                 r_assay_recombined=np.mean(r_k), sd_original=np.std(r_o, ddof=1),
                                 sd_recombined=np.std(r_k, ddof=1), z=z,
                                 power_two_sided=float(norm.cdf(abs(z) - 1.96) + norm.cdf(-abs(z) - 1.96))))
        d = pd.DataFrame([r for r in rows if r["population"] == name])
        print(f"{name}: r0 {d[d.k == 0].r.mean():+.3f}, r(k=4) {d[d.k == 4].r.mean():+.3f}, "
              f"r(k=16) {d[d.k == 16].r.mean():+.3f}, r_LE {d.r_LE.mean():+.3f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", f"distinguish_{args.profile}.csv"), index=False)
        pd.DataFrame(prow).to_csv(os.path.join(ROOT, "results", f"distinguish_power_{args.profile}.csv"), index=False)
    if not rows:
        print("no saved populations")
        return
    df = pd.DataFrame(rows)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                         "savefig.dpi": 150, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, g in df.groupby("population"):
        m = g.groupby("k").r.mean()
        se = g.groupby("k").r.std(ddof=1) / np.sqrt(g.groupby("k").r.count())
        xs = [k + 0.5 for k in m.index]
        ax.errorbar(xs, m.values, yerr=1.96 * se.values, fmt="o-", ms=4, capsize=2, label=name)
        ax.plot([40], [g.r_LE.mean()], "x", color="#52514e")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set(xscale="log", xlabel="generations of female meiosis among hemigenomes (+0.5; x at right = linkage equilibrium)",
           ylabel="r between female and male hemigenome values",
           title="Distinguishing SA (r persists) from linkage/HRI (r decays to r_LE)")
    ax.legend(fontsize=6.5, ncol=2)
    save_png(fig, os.path.join(ROOT, "results", f"fig_distinguish_{args.profile}.png"))


if __name__ == "__main__":
    main()

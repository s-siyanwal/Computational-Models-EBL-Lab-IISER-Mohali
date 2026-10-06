"""Poster (Samant, Moitra, Sinha & Prasad): reproduce every analysis with the
deterministic oracle, and confront each with the zero-leakage IBM."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from .. import config as C
from .. import io
from ..analysis import stats as S
from ..deterministic import poster_model as pm
from ..ibm import runners as R
from ..viz import style as V

P = C.POSTER
log = io.log


def _seed(name):
    return C.seed_for("poster/" + name)


# =========================================================================
# P0  one-locus IaSC polymorphism
# =========================================================================
def one_locus():
    b = P.b_strong
    lo, hi = pm.polymorphism_range(b)
    a_fine = np.linspace(lo - 0.01, hi + 0.01, 400)
    p_formula = np.clip(pm.pstar(a_fine, b), 0, 1)
    # oracle check: iterate the recursion itself
    p = np.full_like(a_fine, 0.5)
    for _ in range(20000):
        p = pm.one_locus_step(p, a_fine, b)
    a_ibm = np.linspace(lo + 0.004, hi - 0.004, 9)
    rows = []
    for N in P.one_locus_N:
        for a in a_ibm:
            w = C.validate_viability(C.poster_viability(a, b, 0, 1))
            out = R.run_haploid([0, N // 2, 0, N - N // 2], w, 0.5, 1500, P.one_locus_reps,
                                _seed(f"ol{N}{a:.4f}"))
            m = out["pA1"][:, 500:].mean(1)       # time average per replicate
            rows.append(dict(N=N, a=a, mean=m.mean(), sd=m.std(ddof=1), median=float(np.median(out["pA1"][:, 500:])),
                             q05=np.quantile(out["pA1"][:, 500:], 0.05),
                             q95=np.quantile(out["pA1"][:, 500:], 0.95),
                             pstar=float(pm.pstar(a, b))))
    df = pd.DataFrame(rows)
    io.save_table(df, "poster_one_locus")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(a_fine, p_formula, color=V.DET, lw=2, label="p* formula (poster)")
    ax.plot(a_fine, p, color=V.MUTED, lw=1, ls=(0, (4, 3)), label="iterated recursion")
    for i, N in enumerate(P.one_locus_N):
        d = df[df.N == N]
        off = (i - 0.5) * 0.0012
        ax.errorbar(d.a + off, d["median"], yerr=V.err(d["median"] - d.q05, d.q95 - d["median"]), fmt="o",
                    color=V.CAT[i], ms=6, capsize=2, lw=1.2, label=f"IBM N={N} (median, 5-95%)")
        ax.scatter(d.a + off, d["mean"], marker="D", s=22, facecolor=V.SURFACE, edgecolor=V.CAT[i], zorder=4,
                   label=f"IBM N={N} mean")
    for x in (lo, hi):
        ax.axvline(x, color=V.AXIS, lw=1)
    ax.set(xlabel="a  (benefit of A1 in males)", ylabel="frequency of A1",
           title="IaSC polymorphism, b = 0.2: oracle vs individual-based populations")
    ax.legend(loc="upper left")
    V.save(fig, io.fig_path("P0_one_locus"))
    big = df[df.N == max(P.one_locus_N)]
    return dict(max_abs_err_largeN=float(np.abs(big["mean"] - big.pstar).max()),
                rmse_by_N={int(N): S.rmse(d["mean"], d.pstar) for N, d in df.groupby("N")},
                iterate_vs_formula=float(np.abs(p - p_formula)[(a_fine > lo + 1e-3) & (a_fine < hi - 1e-3)].max()))


# =========================================================================
# P1  'fraction of parameter space where the fixed point is stable'
# =========================================================================
def stability_heatmaps():
    n = P.heat_n
    panels = [("b = 0.2 (strong)", P.b_strong, np.linspace(0, 1, n), np.linspace(0, 1, n)),
              ("b = 0.02 (weak)", P.b_weak, np.linspace(0, 1, n), np.linspace(0, 1, n)),
              ("small-effect modifiers, b = 0.2", P.b_strong, np.linspace(0, 0.1, n), np.linspace(0.9, 1, n))]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    summary = {}
    for ax, (title, b, k1s, k2s) in zip(axs, panels):
        K1, K2 = np.meshgrid(k1s, k2s)
        F = pm.stable_fraction(b, K1, K2, n_a=P.ar_n, n_r=P.ar_n)
        io.save_arrays("P1_" + title.split()[0] + title.split()[2].strip("(,"), k1=k1s, k2=k2s, frac=F, b=b)
        im = ax.pcolormesh(k1s, k2s, F, cmap=V.SEQ, vmin=0, vmax=1, shading="nearest")
        cs = ax.contour(k1s, k2s, F, levels=[0.25, 0.5, 0.75], colors=[V.SURFACE], linewidths=0.8)
        ax.clabel(cs, fmt="%.2f", fontsize=7)
        ax.set(xlabel="k1 (M1 effect on A1 in females)", ylabel="k2 (M1 effect on A2 in females)", title=title)
        ax.grid(False)
        summary[title] = dict(mean=float(F.mean()),
                              at_k=(float(F[np.argmin(abs(k2s - k2s[n // 2])), np.argmin(abs(k1s - k1s[n // 2]))])))
    fig.colorbar(im, ax=axs, shrink=0.85, label="fraction of (a, r) space where IaSC fixed point is stable")
    fig.suptitle("Linear stability of the IaSC fixed point (x11 = x21 = 0, x12 = p*)  -  poster heatmaps reproduced",
                 color=V.INK, fontsize=12)
    V.save(fig, io.fig_path("P1_stability_heatmaps"))
    # explicit numbers for the poster's representative modifiers
    reps = {f"{k}": dict(b02=float(pm.stable_fraction(0.2, *k, n_a=P.ar_n, n_r=P.ar_n)),
                         b002=float(pm.stable_fraction(0.02, *k, n_a=P.ar_n, n_r=P.ar_n)))
            for k in P.k_pairs}
    summary["representative"] = reps
    return summary


# =========================================================================
# P2  stability maps in (a, r) + IBM invasion probabilities
# =========================================================================
def _ar_grid(b, n):
    lo, hi = pm.polymorphism_range(b)
    A = np.linspace(lo, hi, n + 2)[1:-1]
    Rr = np.linspace(0, 0.5, n + 1)[1:]
    return A, Rr


def ar_maps():
    b = P.b_strong
    A, Rr = _ar_grid(b, P.ar_n)
    AA, RR = np.meshgrid(A, Rr)
    ks = P.k_grid3
    fig, axs = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for i, k2 in enumerate(ks[::-1]):
        for j, k1 in enumerate(ks):
            lam = pm.modifier_block_eigenvalue(AA, b, k1, k2, RR)
            ax = axs[i, j]
            ax.pcolormesh(A, Rr, (lam < 1 + pm.NEUTRAL_TOL).astype(int), cmap=V.STABLE_CMAP,
                          vmin=0, vmax=1, shading="nearest")
            ax.grid(False)
            ax.set_title(f"k1={k1}, k2={k2}", fontsize=9)
            if i == 2:
                ax.set_xlabel("a")
            if j == 0:
                ax.set_ylabel("r")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#eda100"), plt.Rectangle((0, 0), 1, 1, color="#0d366b")]
    fig.legend(handles, ["stable: IaSC persists (M1 cannot invade)", "unstable: M1 invades (resolution)"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Stability of the IaSC fixed point in (a, r), b = 0.2  (poster 3x3 panel)", y=1.05, color=V.INK)
    V.save(fig, io.fig_path("P2a_ar_stability_grid"))

    # ---- IBM invasion experiments on a coarse grid
    # Each IBM population first finds its own IaSC state (burn-in), then M1 is
    # introduced by mutation at frequency m0. Over a fixed horizon we compare
    # the ensemble-mean M1 frequency with the oracle trajectory started from
    # (p*, m0, D=0). Deterministic invasion <=> mean M1 rises above m0.
    a_ibm = np.linspace(A[0] + 0.004, A[-1] - 0.004, P.inv_a)
    rng = np.random.default_rng(_seed("inv-rng"))
    T = P.inv_horizon
    rows = []
    t0 = time.time()
    for k in P.inv_kpairs:
        for a in a_ibm:
            w = C.validate_viability(C.poster_viability(a, b, *k))
            nA1 = R.ibm_iasc_burn_in(P.N_inv, w, P.reps_inv, _seed(f"burn{k}{a:.4f}"))
            for r in P.inv_r:
                m1 = int(round(P.inv_m0 * P.N_inv))
                init = R.introduce_modifier(nA1, P.N_inv, m1, rng)
                out = R.run_haploid(init, w, r, T, P.reps_inv, _seed(f"inv{k}{a:.4f}{r}"))
                m_T = out["pM1"][:, -1]
                m0_emp = init[:, [0, 2]].sum(1) / P.N_inv
                ratio = m_T.mean() / m0_emp.mean()
                se = m_T.std(ddof=1) / np.sqrt(len(m_T)) / m0_emp.mean()
                _, _, traj = pm.iterate(pm.from_allele_freqs(pm.pstar(a, b), P.inv_m0, 0.0), a, b, *k, r,
                                        n_gen=T, record=True)
                det_T = float(pm.summaries(traj[-1])[1])
                lam = float(pm.modifier_block_eigenvalue(a, b, *k, r))
                rows.append(dict(k1=k[0], k2=k[1], a=a, r=r, lam=lam, det_invades=lam > 1 + pm.NEUTRAL_TOL,
                                 n=P.reps_inv, ibm_ratio=float(ratio), ibm_ratio_lo=float(ratio - 1.96 * se),
                                 ibm_ratio_hi=float(ratio + 1.96 * se), det_ratio=det_T / P.inv_m0,
                                 frac_increased=float(np.mean(m_T > m0_emp)),
                                 A_polymorphic_at_intro=float(np.mean((nA1 > 0) & (nA1 < P.N_inv)))))
        log.info("  P2 IBM %s done (%.0fs)", k, time.time() - t0)
    df = pd.DataFrame(rows)
    df["verdict"] = np.where(df.ibm_ratio_lo > 1, "IBM: spreads",
                             np.where(df.ibm_ratio_hi < 1, "IBM: declines", "IBM: unresolved"))
    # primary: direction of the oracle trajectory from the same 2% dose
    # (finite dose can cross a bistability threshold even where lambda < 1);
    # secondary: the linear-stability prediction (lambda > 1)
    det_up = df.det_ratio > 1
    df["agree"] = (((df.verdict == "IBM: spreads") & det_up) | ((df.verdict == "IBM: declines") & ~det_up))
    df["agree_lambda"] = (((df.verdict == "IBM: spreads") & df.det_invades)
                          | ((df.verdict == "IBM: declines") & ~df.det_invades))
    io.save_table(df, "poster_invasion_ibm")

    fig, axs = plt.subplots(1, len(P.inv_kpairs) + 1, figsize=(5 * (len(P.inv_kpairs) + 1), 4.6))
    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    for ax, k in zip(axs[:-1], P.inv_kpairs):
        lam = pm.modifier_block_eigenvalue(AA, b, *k, RR)
        ax.pcolormesh(A, Rr, (lam < 1).astype(int),
                      cmap=plt.matplotlib.colors.ListedColormap(["#fde8e8", "#e3edf9"]),
                      shading="nearest", vmin=0, vmax=1)
        ax.contour(A, Rr, lam, levels=[1.0], colors=[V.INK], linewidths=1.6)
        d = df[(df.k1 == k[0]) & (df.k2 == k[1])]
        sc = ax.scatter(d.a, d.r, c=np.log2(d.ibm_ratio.clip(1 / 8, 8)), cmap=V.DIV, norm=norm, s=170,
                        edgecolor=[V.SURFACE if g else V.INK for g in d.agree], linewidth=1.8, zorder=3)
        ax.grid(False)
        ax.set(title=V.klabel(k), xlabel="a", ylabel="r")
    fig.colorbar(sc, ax=axs[-2], label=f"IBM log2(mean M1 at t={T} / m0)")
    ax = axs[-1]
    for k in P.inv_kpairs:
        d = df[(df.k1 == k[0]) & (df.k2 == k[1])]
        ax.errorbar(d.det_ratio, d.ibm_ratio, yerr=V.err(d.ibm_ratio - d.ibm_ratio_lo, d.ibm_ratio_hi - d.ibm_ratio),
                    fmt="o", color=V.K_COLORS[k], ms=5, capsize=0, lw=0.8, label=V.klabel(k))
    lim = [max(1e-3, min(df.det_ratio.min(), df.ibm_ratio.min()) * 0.8),
           max(df.det_ratio.max(), df.ibm_ratio.max()) * 1.2]
    ax.plot(lim, lim, color=V.INK, lw=1, ls="--", label="1:1")
    ax.set(xscale="log", yscale="log", xlabel=f"oracle M1(t={T}) / m0", ylabel=f"IBM mean M1(t={T}) / m0",
           title="Early spread of the modifier: IBM vs oracle")
    ax.legend(fontsize=7.5)
    fig.suptitle(f"Invasion of a {P.inv_m0:.0%} modifier. Black line: deterministic boundary (blue side: M1 invades). "
                 f"Dots: IBM, N={P.N_inv}, {P.reps_inv} pops (dark ring = IBM contradicts oracle)",
                 color=V.INK, fontsize=10.5)
    V.save(fig, io.fig_path("P2b_invasion_ibm_vs_boundary"))
    decided = df[df.verdict != "IBM: unresolved"]
    return dict(verdict_counts=df.verdict.value_counts().to_dict(), n_cells=len(df),
                agreement_among_decided=float(decided.agree.mean()) if len(decided) else None,
                agreement_with_linear_stability=float(decided.agree_lambda.mean()) if len(decided) else None,
                log_ratio_corr=float(np.corrcoef(np.log(df.det_ratio), np.log(df.ibm_ratio.clip(1e-3)))[0, 1]),
                disagreements=df[(df.verdict != "IBM: unresolved") & ~df.agree][
                    ["k1", "k2", "a", "r", "lam", "ibm_ratio", "det_ratio"]].to_dict("records"))


# =========================================================================
# P2c  persistence of the IaSC polymorphism in finite populations
# =========================================================================
def persistence():
    b = P.b_strong
    lo, hi = pm.polymorphism_range(b)
    a_vals = np.linspace(lo + 0.002, hi - 0.002, P.persist_a)
    # deterministic restoring eigenvalue of the one-locus map at p*
    h = 1e-7
    a_fine = np.linspace(lo + 1e-4, hi - 1e-4, 300)
    ps = pm.pstar(a_fine, b)
    mu = (pm.one_locus_step(ps + h, a_fine, b) - pm.one_locus_step(ps - h, a_fine, b)) / (2 * h)
    rows = []
    for N in P.persist_N:
        for a in a_vals:
            w = C.validate_viability(C.poster_viability(a, b, 0, 1))
            # start at 50% A1 (no use of p*) and let selection + drift act
            out = R.run_haploid([0, N // 2, 0, N - N // 2], w, 0.5, P.persist_gens, P.persist_reps,
                                _seed(f"pers{N}{a:.4f}"))
            pA = out["pA1"]
            poly = (pA > 0) & (pA < 1)
            lost_at = np.where(poly.all(1), P.persist_gens, np.argmin(poly, 1))
            rows.append(dict(N=N, a=a, frac_poly_end=float(poly[:, -1].mean()),
                             mean_time_poly=float(lost_at.mean())))
    df = pd.DataFrame(rows)
    io.save_table(df, "poster_persistence")
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axs[0].plot(a_fine, mu, color=V.INK)
    axs[0].axhline(1, color=V.AXIS)
    axs[0].set(xlabel="a", ylabel="restoring eigenvalue dp'/dp at p*",
               title="Oracle: polymorphism stable (< 1) across the range,\nbut only weakly near its edges")
    for i, N in enumerate(P.persist_N):
        d = df[df.N == N]
        axs[1].plot(d.a, d.frac_poly_end, "o-", color=V.CAT[i], label=f"IBM N={N}")
    axs[1].set(xlabel="a", ylabel=f"fraction still polymorphic at t={P.persist_gens}",
               ylim=(-0.03, 1.03), title="IBM: drift erodes IaSC near the edges of the range")
    axs[1].legend()
    V.save(fig, io.fig_path("P2c_persistence"))
    return dict(table=df.to_dict("records"), mu_min=float(mu.min()), mu_max=float(mu.max()))


# =========================================================================
# P3  bifurcation diagrams (r = 0.1, b = 0.2)
# =========================================================================
def bifurcation():
    b, r = P.b_strong, P.r_bif
    lo, hi = pm.polymorphism_range(b)
    a_det = np.linspace(0.16, 0.26, 401)
    det = {}
    for k in P.k_pairs:
        # near the fixed point: p* with a tiny dose of M1 (D = 0)
        ps = np.clip(pm.pstar(a_det, b), 1e-9, 1 - 1e-9)
        x0n = pm.from_allele_freqs(ps, P.near_eps, 0.0)
        xn, _ = pm.fate_fast(x0n, a_det, b, *k, r, max_gen=40000)
        x0a = pm.from_allele_freqs(np.full_like(a_det, P.away_ic[0]), P.away_ic[1], P.away_ic[2])
        xa, _ = pm.fate_fast(x0a, a_det, b, *k, r, max_gen=40000)
        det[k] = dict(near=pm.summaries(xn)[:2], away=pm.summaries(xa)[:2])
    # IBM
    a_ibm = np.linspace(0.162, 0.255, P.bif_a)
    rng = np.random.default_rng(_seed("bif-rng"))
    rows = []
    for k in P.k_pairs:
        for a in a_ibm:
            w = C.validate_viability(C.poster_viability(a, b, *k))
            nA1 = R.ibm_iasc_burn_in(P.N_bif, w, P.bif_reps, _seed(f"bburn{k}{a:.4f}"))
            init_near = R.introduce_modifier(nA1, P.N_bif, int(round(P.near_m0_ibm * P.N_bif)), rng)
            init_away = R.founders_haploid(P.N_bif, P.away_ic[0], P.away_ic[1], rng)
            for kind, init in (("near", init_near), ("away", init_away)):
                out = R.run_haploid(init, w, r, P.bif_gens, P.bif_reps, _seed(f"bif{kind}{k}{a:.4f}"))
                rows.append(dict(k1=k[0], k2=k[1], a=a, kind=kind,
                                 A1_mean=out["pA1"][:, -200:].mean(), M1_mean=out["pM1"][:, -1].mean(),
                                 A1_q=np.quantile(out["pA1"][:, -1], [0.1, 0.9]).tolist(),
                                 P_resolved=float(np.mean(out["pM1"][:, -1] > 0.5))))
    df = pd.DataFrame(rows)
    io.save_table(df.drop(columns=["A1_q"]), "poster_bifurcation_ibm")
    np.savez_compressed(io.DATA + "/poster_bifurcation_det.npz", a=a_det,
                        **{f"{kind}_{k[0]}_{k[1]}_{v}": det[k][kind][i] for k in P.k_pairs
                           for kind in ("near", "away") for i, v in enumerate(("A1", "M1"))})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for row, kind in enumerate(("near", "away")):
        for col, (var, lab) in enumerate((("A1", "equilibrium frequency of A1"), ("M1", "equilibrium frequency of M1"))):
            ax = axs[row, col]
            for k in P.k_pairs:
                c = V.K_COLORS[k]
                ax.plot(a_det, det[k][kind][col], color=c, lw=1.6, label=V.klabel(k))
                d = df[(df.k1 == k[0]) & (df.k2 == k[1]) & (df.kind == kind)]
                ax.scatter(d.a, d[f"{var}_mean"], color=c, s=26, edgecolor=V.SURFACE, linewidth=1.2, zorder=3)
            ax.axvline(lo, color=V.AXIS, lw=1)
            ax.axvline(hi, color=V.AXIS, lw=1)
            ax.set(ylabel=lab, ylim=(-0.03, 1.03))
            ax.set_title(("starting near the fixed point" if kind == "near" else
                          f"starting away (pA1={P.away_ic[0]}, pM1={P.away_ic[1]}, D=0)"), fontsize=10)
            if row == 1:
                ax.set_xlabel("a")
    axs[0, 1].legend(loc="center right", fontsize=7.5)
    axs[0, 0].add_artist(axs[0, 0].legend([Line2D([], [], color=V.INK, lw=1.6),
                                           Line2D([], [], color=V.INK, marker="o", lw=0)],
                                          ["deterministic", f"IBM mean (N={P.N_bif})"], loc="upper left"))
    fig.suptitle("Bifurcation diagrams, r = 0.1, b = 0.2  (lines: oracle; dots: individual-based ensemble means)",
                 color=V.INK)
    V.save(fig, io.fig_path("P3_bifurcation"))
    # jump positions
    jumps = {}
    for k in P.k_pairs:
        m = det[k]["near"][1]
        idx = np.where(m > 0.5)[0]
        idx = idx[(a_det[idx] > lo) & (a_det[idx] < hi)]
        jumps[V.klabel(k)] = float(a_det[idx[0]]) if idx.size else None
    return dict(det_jump_near=jumps)


# =========================================================================
# P4  minimum modifier frequency for resolution + IBM threshold curves
# =========================================================================
def thresholds():
    b, r = P.b_strong, P.r_bif
    lo, hi = pm.polymorphism_range(b)
    a_det = np.linspace(lo + 1e-3, hi - 1e-3, 30)
    mins = {k: pm.min_modifier_frequency(a_det, b, *k, r) for k in P.k_pairs if k != (0.0, 0.5)}
    io.save_arrays("poster_min_modifier", a=a_det, **{V.klabel(k): v for k, v in mins.items()})
    rows = []
    rng = np.random.default_rng(_seed("thr-rng"))
    k = P.thr_k
    for a in P.thr_a:
        w = C.validate_viability(C.poster_viability(a, b, *k))
        thr_det = float(pm.min_modifier_frequency(a, b, *k, r)[0])
        for N in P.thr_N:
            nA1 = R.ibm_iasc_burn_in(N, w, P.thr_reps, _seed(f"tburn{a}{N}"))
            for m0 in P.thr_m0:
                init = R.introduce_modifier(nA1, N, int(round(m0 * N)), rng)
                out = R.run_haploid(init, w, r, P.thr_gens, P.thr_reps, _seed(f"thr{a}{N}{m0}"),
                                    stop_on_m_absorb=True)
                kres = int(np.sum(out["pM1"][:, -1] > 0.5))
                p_, l_, h_ = S.wilson(kres, P.thr_reps)
                rows.append(dict(a=a, N=N, m0=m0, k=kres, n=P.thr_reps, p=float(p_), lo=float(l_), hi=float(h_),
                                 det_threshold=thr_det))
    df = pd.DataFrame(rows)
    io.save_table(df, "poster_threshold_ibm")
    fits = {}
    for (a, N), d in df.groupby(["a", "N"]):
        x50, s = S.fit_logistic(d.m0, d.k, d.n)
        fits[f"a={a}, N={N}"] = dict(x50=x50, width=s, det=float(d.det_threshold.iloc[0]))

    fig, axs = plt.subplots(1, 1 + len(P.thr_a), figsize=(5.2 * (1 + len(P.thr_a)), 4.2))
    ax = axs[0]
    for kk, v in mins.items():
        ax.plot(a_det, v, color=V.K_COLORS[kk], label=V.klabel(kk))
    ax.set(xlabel="a", ylabel="min. initial M1 frequency for resolution", title="Deterministic threshold (D = 0, r = 0.1)")
    ax.legend(fontsize=7)
    for ax, a in zip(axs[1:], P.thr_a):
        d0 = df[df.a == a]
        for i, N in enumerate(P.thr_N):
            d = d0[d0.N == N]
            ax.errorbar(d.m0, d.p, yerr=V.err(d.p - d.lo, d.hi - d.p), fmt="o-", color=V.CAT[i], lw=1.4, ms=5,
                        capsize=2, label=f"IBM N={N}")
        ax.axvline(d0.det_threshold.iloc[0], color=V.INK, lw=1.6, ls="--", label="deterministic threshold")
        ax.set(xlabel="initial M1 frequency", ylabel="P(resolution)", ylim=(-0.03, 1.03),
               title=f"{V.klabel(k)}, a = {a}")
        ax.legend(fontsize=7.5)
    fig.suptitle("Bistability: a finite dose of M1 is needed when the IaSC point is stable", color=V.INK)
    V.save(fig, io.fig_path("P4_thresholds"))
    return dict(logistic_fits=fits)


# =========================================================================
# P5  basins: fraction of random initial conditions ending at the IaSC point
# =========================================================================
def basins():
    b, r = P.b_strong, P.r_bif
    lo, hi = pm.polymorphism_range(b)
    rng = np.random.default_rng(_seed("basin"))
    ics = rng.dirichlet(np.ones(4), size=P.basin_det_n)
    a_det = np.linspace(0.16, 0.26, 41)
    det = {}
    X0 = np.broadcast_to(ics, (a_det.size,) + ics.shape)
    acol = a_det[:, None]
    for k in P.k_pairs:
        x, _ = pm.fate_fast(X0, acol, b, *k, r, max_gen=20000)
        lab = np.where((acol > lo) & (acol < hi), pm.classify(x, acol, b), 2)
        det[k] = np.mean(lab == 0, axis=1)
        log.info("  P5 deterministic basins %s done", k)
    a_ibm = np.linspace(lo + 0.004, hi - 0.004, P.basin_a)
    rows = []
    ics_ibm = ics[: P.basin_ibm_n]
    for k in P.basin_kpairs:
        for a in a_ibm:
            w = C.validate_viability(C.poster_viability(a, b, *k))
            init = np.array([R.founders_from_haplotypes(P.basin_N, x) for x in ics_ibm])
            out = R.run_haploid(init, w, r, P.basin_gens, len(ics_ibm), _seed(f"bas{k}{a:.4f}"))
            pM1 = out["pM1"][:, -1]
            xd, _ = pm.fate_fast(ics_ibm, a, b, *k, r, max_gen=30000)
            lab_det = pm.classify(xd, a, b)
            lab_ibm = np.where(pM1 == 0, 0, np.where(pM1 == 1, 1, 2))
            rows.append(dict(k1=k[0], k2=k[1], a=a, ibm_frac_iasc=float(np.mean(lab_ibm == 0)),
                             det_frac_iasc_same_ics=float(np.mean(lab_det == 0)),
                             per_ic_agreement=float(np.mean(lab_ibm == lab_det)), n=len(ics_ibm)))
    df = pd.DataFrame(rows)
    io.save_table(df, "poster_basins_ibm")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for k in P.k_pairs:
        ax.plot(a_det, det[k], color=V.K_COLORS[k], lw=1.6, label=V.klabel(k))
    for k in P.basin_kpairs:
        d = df[(df.k1 == k[0]) & (df.k2 == k[1])]
        p_, l_, h_ = S.wilson(d.ibm_frac_iasc * d.n, d.n)
        ax.errorbar(d.a, p_, yerr=V.err(p_ - l_, h_ - p_), fmt="o", color=V.K_COLORS[k], mec=V.SURFACE, ms=7,
                    capsize=2, zorder=3)
    ax.set(xlabel="a", ylabel="fraction of initial conditions reaching IaSC point (M1 lost)",
           title=f"Basin of the IaSC equilibrium, b = 0.2, r = 0.1  (dots: IBM, N={P.basin_N})")
    ax.legend(fontsize=7.5, loc="upper right")
    V.save(fig, io.fig_path("P5_basins"))
    return dict(per_ic_agreement=df.groupby(["k1", "k2"]).per_ic_agreement.mean().to_dict(),
                det_peak={V.klabel(k): float(det[k].max()) for k in P.k_pairs})


# =========================================================================
# P6  convergence of the IBM ensemble to the oracle as N grows
# =========================================================================
def convergence():
    b, r, k, a = P.b_strong, P.r_bif, P.conv_k, P.conv_a
    x0 = pm.from_allele_freqs(*P.conv_ic)
    _, _, traj = pm.iterate(x0, a, b, *k, r, n_gen=P.conv_gens, record=True)
    pA_det, pM_det, _ = pm.summaries(traj)
    w = C.validate_viability(C.poster_viability(a, b, *k))
    rows, means = [], {}
    rng = np.random.default_rng(_seed("conv"))
    for N in P.conv_N:
        init = R.founders_haploid(N, P.conv_ic[0], P.conv_ic[1], rng)
        out = R.run_haploid(init, w, r, P.conv_gens, P.conv_reps, _seed(f"conv{N}"))
        mA, mM = out["pA1"].mean(0), out["pM1"].mean(0)
        means[N] = (mA, mM, out["pA1"].std(0), out["pM1"].std(0))
        rows.append(dict(N=N, rmse_A1=S.rmse(mA, pA_det), rmse_M1=S.rmse(mM, pM_det),
                         sd_A1_t100=float(out["pA1"][:, 100].std()), sd_M1_t100=float(out["pM1"][:, 100].std())))
    df = pd.DataFrame(rows)
    io.save_table(df, "poster_convergence")
    slope = S.loglog_slope(df.N, df.sd_M1_t100)
    slope_rmse = S.loglog_slope(df.N, df.rmse_M1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    t = np.arange(P.conv_gens + 1)
    ax = axs[0]
    for i, N in enumerate([P.conv_N[0], P.conv_N[len(P.conv_N) // 2], P.conv_N[-1]]):
        mA, mM, sA, sM = means[N]
        ax.fill_between(t, mM - sM, mM + sM, color=V.CAT[i], alpha=0.18, lw=0)
        ax.plot(t, mM, color=V.CAT[i], lw=1.4, label=f"IBM N={N} (mean ± sd)")
    ax.plot(t, pM_det, color=V.INK, lw=2, ls="--", label="deterministic")
    ax.set(xlabel="generation", ylabel="frequency of M1", title=f"M1 sweep, {V.klabel(k)}, a={a}, r={r}")
    ax.legend(fontsize=7.5)
    ax = axs[1]
    ax.loglog(df.N, df.rmse_M1, "o-", color=V.CAT[0], label="RMSE(ensemble mean, oracle)")
    ax.loglog(df.N, df.sd_M1_t100, "s-", color=V.CAT[1], label="between-population SD (t=100)")
    ref = df.sd_M1_t100.iloc[0] * np.sqrt(df.N.iloc[0] / df.N)
    ax.loglog(df.N, ref, color=V.MUTED, ls=":", lw=1.2, label="N^-1/2 reference")
    ax.set(xlabel="population size N", ylabel="frequency error",
           title=f"Drift noise shrinks as N^{slope[0]:.2f} (95% CI {slope[1]:.2f} to {slope[2]:.2f})")
    ax.legend(fontsize=7.5)
    V.save(fig, io.fig_path("P6_convergence"))
    big = df[df.N >= 1000]
    return dict(sd_slope=slope, rmse_slope=slope_rmse, sd_slope_N_ge_1000=S.loglog_slope(big.N, big.sd_M1_t100),
                table=df.to_dict("records"))


# =========================================================================
# Animation: IBM populations in the (pA1, pM1) plane with oracle trajectories
# =========================================================================
def animation():
    b, r, k, a = P.b_strong, P.r_bif, P.anim_k, P.anim_a
    w = C.validate_viability(C.poster_viability(a, b, *k))
    grid = [(pa, pmo) for pa in (0.2, 0.5, 0.8) for pmo in (0.05, 0.15, 0.3, 0.5, 0.7)]
    rng = np.random.default_rng(_seed("anim"))
    reps = 3
    traj_ibm, traj_det, fates = [], [], []
    for i, (pa, pmo) in enumerate(grid):
        init = R.founders_haploid(P.anim_N, pa, pmo, rng)
        out = R.run_haploid(init, w, r, P.anim_gens, reps, _seed(f"anim{i}"))
        traj_ibm.append(np.stack([out["pA1"], out["pM1"]], -1))
        _, _, tr = pm.iterate(pm.from_allele_freqs(pa, pmo, 0), a, b, *k, r, n_gen=P.anim_gens, record=True)
        pa_d, pm_d, _ = pm.summaries(tr)
        traj_det.append(np.stack([pa_d, pm_d], -1))
        fates.append(pm_d[-1] > 0.5)
    traj_ibm = np.array(traj_ibm)        # (ic, rep, t, 2)
    traj_det = np.array(traj_det)        # (ic, t, 2)
    ps = float(pm.pstar(a, b))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for d, f in zip(traj_det, fates):
        ax.plot(d[:, 0], d[:, 1], color=V.CAT[1] if f else V.CAT[0], lw=1, alpha=0.55)
    ax.plot([ps], [0], marker="*", ms=16, color=V.CAT[0], mec=V.INK, zorder=5)
    ax.plot([1], [1], marker="*", ms=16, color=V.CAT[1], mec=V.INK, zorder=5)
    ax.annotate("IaSC point (p*, M1 lost)", (ps, 0), xytext=(ps - 0.25, 0.08), fontsize=8, color=V.INK2)
    ax.annotate("resolution (A1, M1 fixed)", (1, 1), xytext=(0.55, 0.95), fontsize=8, color=V.INK2)
    cols = np.repeat([V.CAT[1] if f else V.CAT[0] for f in fates], reps)
    pts = traj_ibm[:, :, 0, :].reshape(-1, 2)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=30, edgecolor=V.SURFACE, linewidth=1, zorder=4)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, color=V.INK)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="frequency of A1", ylabel="frequency of M1",
           title=f"IBM populations (dots, N={P.anim_N}) on oracle trajectories (lines)\n{V.klabel(k)}, a={a}, r={r}: bistable")
    frames = list(range(0, P.anim_gens + 1, max(1, P.anim_gens // 150)))

    def upd(t):
        sc.set_offsets(traj_ibm[:, :, t, :].reshape(-1, 2))
        txt.set_text(f"generation {t}")
        return sc, txt

    anim = FuncAnimation(fig, upd, frames=frames, blit=False)
    path = io.anim_path("poster_bistability")
    anim.save(path, writer=PillowWriter(fps=15))
    upd(frames[-1])
    V.save(fig, io.fig_path("P7_animation_lastframe"))
    ibm_res = (traj_ibm[:, :, -1, 1] > 0.5)
    agree = float(np.mean(ibm_res == np.array(fates)[:, None]))
    return dict(gif=path, per_population_fate_agreement=agree)


ALL = [("P0 one-locus polymorphism", one_locus), ("P1 stability heatmaps", stability_heatmaps),
       ("P2 (a,r) maps + IBM invasion", ar_maps), ("P2c IaSC persistence", persistence), ("P3 bifurcation", bifurcation),
       ("P4 thresholds", thresholds), ("P5 basins", basins), ("P6 convergence", convergence),
       ("P7 animation", animation)]

"""Parsons (1961) / Owen (1953): initial progress of new genes with
sex-differential viabilities and with sex linkage; Owen's bistability."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from .. import config as C
from .. import io
from ..analysis import stats as S
from ..deterministic import owen_parsons as op
from ..ibm import runners as R
from ..viz import style as V

O = C.OWEN
log = io.log


def _seed(name):
    return C.seed_for("owen/" + name)


def _direction(final, f0, det_ratio):
    """Mean frequency at the horizon relative to the founding dose, with a
    95% CI; compared with the oracle's ratio from the same dose."""
    final = final[np.isfinite(final)]
    ratio = final.mean() / f0
    se = final.std(ddof=1) / np.sqrt(len(final)) / f0
    lo, hi = ratio - 1.96 * se, ratio + 1.96 * se
    verdict = "IBM: spreads" if lo > 1 else ("IBM: declines" if hi < 1 else "IBM: unresolved")
    agree = (verdict == "IBM: spreads" and det_ratio > 1) or (verdict == "IBM: declines" and det_ratio < 1)
    return dict(ibm_ratio=float(ratio), ibm_ratio_lo=float(lo), ibm_ratio_hi=float(hi), det_ratio=float(det_ratio),
                verdict=verdict, agree=bool(agree))


def _ratio_panel(ax, df, title):
    for v, c in ((True, V.CAT[0]), (False, V.CAT[1])):
        d = df[df.det_invades == v]
        ax.errorbar(d.det_ratio, d.ibm_ratio, yerr=V.err(d.ibm_ratio - d.ibm_ratio_lo, d.ibm_ratio_hi - d.ibm_ratio),
                    fmt="o", ms=5, color=c, lw=0.8, label="oracle: invades" if v else "oracle: cannot invade")
    lim = [max(1e-3, min(df.det_ratio.min(), df.ibm_ratio.clip(lower=1e-3).min()) * 0.6),
           max(df.det_ratio.max(), df.ibm_ratio.max()) * 1.6]
    ax.plot(lim, lim, color=V.INK, ls="--", lw=1, label="1:1")
    ax.axhline(1, color=V.AXIS, lw=1)
    ax.axvline(1, color=V.AXIS, lw=1)
    ax.set(xscale="log", yscale="log", xlabel="oracle: p(T) / p(0)", ylabel="IBM: mean p(T) / p(0)", title=title,
           xlim=lim, ylim=lim)
    ax.legend(fontsize=7.5)


def _established(p_zyg, init_freq):
    """IBM outcome: allele has clearly increased (>= 5x its founding
    frequency) at the end of the run."""
    return p_zyg[:, -1] >= 5 * init_freq


# =========================================================================
# O1  autosomal invasion map: h1/b1 + h2/b2 > 2
# =========================================================================
def autosomal_invasion():
    g = np.linspace(O.ratio_lo, O.ratio_hi, O.grid_n)
    rows = []
    f0 = O.init_copies / (2 * O.N)
    for x in g:          # x = h1/b1
        for y in g:      # y = h2/b2
            # b = 1, A dominant (a = h): only h/b matters for invasion
            w = C.validate_viability(C.owen_viability(x, x, 1.0, y, y, 1.0))
            out = R.run_diploid(O.N, O.init_copies, 0, w, O.gens, O.reps, _seed(f"inv{x:.3f}{y:.3f}"))
            k = int(_established(out["p_zyg"], f0).sum())
            p_, l_, h_ = S.wilson(k, O.reps)
            lam = float(op.autosomal_invasion_eigenvalue(x, 1, y, 1))
            d1, d2 = op.iterate(op.autosomal_step, f0, f0, (x, x, 1.0, y, y, 1.0), n_gen=O.gens)
            rows.append(dict(h1_b1=x, h2_b2=y, lam=lam, det_invades=bool(op.autosomal_invades(x, 1, y, 1)),
                             k=k, n=O.reps, p=float(p_), lo=float(l_), hi=float(h_),
                             **_direction(out["p_zyg"][:, -1], f0, float((d1 + d2) / 2) / f0)))
    df = pd.DataFrame(rows)
    io.save_table(df, "owen_autosomal_invasion")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    Pm = df.pivot(index="h2_b2", columns="h1_b1", values="p")
    im = ax.pcolormesh(Pm.columns, Pm.index, Pm.values, cmap=V.DIV, vmin=0, vmax=1, shading="nearest")
    xx = np.linspace(O.ratio_lo, O.ratio_hi, 50)
    ax.plot(xx, 2 - xx, color=V.INK, lw=2, label="Parsons: h1/b1 + h2/b2 = 2")
    ax.set(xlim=(O.ratio_lo, O.ratio_hi), ylim=(O.ratio_lo, O.ratio_hi),
           xlabel="h1/b1  (heterozygote advantage in females)", ylabel="h2/b2  (in males)",
           title=f"IBM establishment probability (N={O.N}, {O.init_copies} founding copies)")
    ax.grid(False)
    ax.legend(loc="lower left")
    fig.colorbar(im, ax=ax, label="P(new allele established)")
    _ratio_panel(axs[1], df, f"Spread over {O.gens} generations: IBM vs oracle")
    V.save(fig, io.fig_path("O1_autosomal_invasion"))
    decided = df[df.verdict != "IBM: unresolved"]
    return dict(verdicts=df.verdict.value_counts().to_dict(), agreement_among_decided=float(decided.agree.mean()),
                lambda_vs_establishment_corr=float(np.corrcoef(df.lam, df.p)[0, 1]))


# =========================================================================
# O2  Parsons' worked example: beta1 = 0.05, beta2 = -0.02
# =========================================================================
def worked_example():
    a1, h1, b1, a2, h2, b2 = O.example
    lam = float(op.autosomal_invasion_eigenvalue(h1, b1, h2, b2))
    T = 1500
    traj = op.iterate(op.autosomal_step, 0.01, 0.01, O.example, n_gen=T, record=True)
    w = C.validate_viability(C.owen_viability(*O.example))
    N = 4000
    out = R.run_diploid(N, int(0.02 * N), 0, w, T, 64, _seed("example"))
    pf = out["p_f"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    q = np.nanquantile(pf, [0.1, 0.5, 0.9], axis=0)
    t = np.arange(T + 1)
    ax.fill_between(t, q[0], q[2], color=V.CAT[0], alpha=0.2, lw=0, label="IBM 10-90% (64 pops)")
    ax.plot(t, q[1], color=V.CAT[0], lw=1.5, label="IBM median")
    ax.plot(t, traj[:, 0], color=V.INK, lw=2, ls="--", label="deterministic p1")
    eq = op.autosomal_equilibria(*O.example)
    for e in eq:
        ax.axhline(e["p1"], color=V.MUTED, lw=1, ls=":")
    ax.set(xlabel="generation", ylabel="freq. of new allele A in females",
           title=f"Parsons' example: female het. advantage 5%, male het. disadvantage 2%  (lambda={lam:.4f})")
    ax.legend(fontsize=8)
    V.save(fig, io.fig_path("O2_parsons_example"))
    return dict(lambda_=lam, equilibria=[(e["p1"], e["p2"], e["stable"]) for e in eq],
                ibm_final_median=float(q[1, -1]), det_final=float(traj[-1, 0]),
                ibm_fraction_increased=float(np.mean(out["p_zyg"][:, -1] > 0.02)))


# =========================================================================
# O3  X-linked invasion: h1 (a2 + b2) > 2 b1 b2  ~  2 beta1 + beta2 - alpha2 > 0
# =========================================================================
def xlinked_invasion():
    betas = np.linspace(-0.04, 0.06, O.grid_n)
    alphas = np.linspace(-0.06, 0.1, O.grid_n)
    rows = []
    N = O.N
    f0 = O.init_copies / (1.5 * N)
    for be1 in betas:
        for al2 in alphas:
            h1, b1, a2, b2, a1 = 1.0, 1 - be1, 1 - al2, 1.0, 1.0
            w = C.validate_viability(C.xlinked_viability(a1, h1, b1, a2, b2))
            out = R.run_xlinked(N, O.init_copies, w, O.gens, O.reps, _seed(f"x{be1:.3f}{al2:.3f}"))
            pall = (2 * out["p_f"] + out["p_m"]) / 3
            k = int(np.sum(pall[:, -1] >= 5 * f0))
            p_, l_, h_ = S.wilson(k, O.reps)
            par = (a1, h1, b1, a2, b2)
            q0 = np.nanmean(pall[:, 0])
            d1, d2 = op.iterate(op.sexlinked_step, q0, q0, par, n_gen=O.gens)
            rows.append(dict(beta1=be1, alpha2=al2, lam=float(op.sexlinked_invasion_eigenvalue(h1, b1, a2, b2)),
                             det_invades=bool(op.sexlinked_invades(h1, b1, a2, b2)),
                             approx_invades=bool(2 * be1 - al2 > 0), p=float(p_), lo=float(l_), hi=float(h_), n=O.reps,
                             **_direction(pall[:, -1], q0, float((2 * d1 + d2) / 3) / q0)))
    df = pd.DataFrame(rows)
    io.save_table(df, "owen_xlinked_invasion")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axs[0]
    Pm = df.pivot(index="alpha2", columns="beta1", values="p")
    im = ax.pcolormesh(Pm.columns, Pm.index, Pm.values, cmap=V.DIV, vmin=0, vmax=1, shading="nearest")
    bb = np.linspace(betas[0], betas[-1], 100)
    # exact h1(a2+b2) = 2 b1 b2 with h1 = b2 = 1 reduces to alpha2 = 2 beta1 = Parsons (3.6)
    ax.plot(bb, 2 * bb, color=V.INK, lw=2, label="Parsons (3.6): alpha2 = 2 beta1 (exact here)")
    ax.set(xlabel="beta1  (female heterozygote advantage)", ylabel="alpha2  (cost of A in hemizygous males)",
           title=f"X-linked locus: IBM establishment probability (N={N})",
           xlim=(betas[0], betas[-1]), ylim=(alphas[0], alphas[-1]))
    ax.grid(False)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.colorbar(im, ax=ax, label="P(established)")
    _ratio_panel(axs[1], df, f"X-linked spread over {O.gens} generations")
    V.save(fig, io.fig_path("O3_xlinked_invasion"))
    decided = df[df.verdict != "IBM: unresolved"]
    return dict(verdicts=df.verdict.value_counts().to_dict(), agreement_among_decided=float(decided.agree.mean()),
                approx_vs_exact_disagreements=int((df.det_invades != df.approx_invades).sum()))


# =========================================================================
# O4  Owen's two stable equilibria: phase portrait, IBM basins, animation
# =========================================================================
def owen_bistability():
    par = O.bistable
    eq = op.autosomal_equilibria(*par)
    # flow field
    g = np.linspace(0.005, 0.995, 40)
    P1, P2 = np.meshgrid(g, g)
    N1, N2 = op.autosomal_step(P1, P2, *par)
    U, Vv = N1 - P1, N2 - P2
    # deterministic basin map (fine)
    gb = np.linspace(0.005, 0.995, 200)
    B1, B2 = np.meshgrid(gb, gb)
    f1, f2 = op.iterate(op.autosomal_step, B1, B2, par, n_gen=3000)
    stable = [e for e in eq if e["stable"]]
    lo_eq = min(stable, key=lambda e: e["p1"])
    basin_hi = (np.hypot(f1 - lo_eq["p1"], f2 - lo_eq["p2"]) > 0.05)
    # IBM from a grid of starting points
    starts = np.linspace(0.1, 0.9, O.phase_ics)
    rows, trajs = [], []
    w = C.validate_viability(C.owen_viability(*par))
    for p0 in starts:
        for q0 in starts:
            # founders: female/male gene pools approximated by drawing every
            # individual's two genes with prob (p0+q0)/2 -- sexes then mix
            # through mating; the IBM is told only the founding allele count
            copies = int(round((p0 + q0) / 2 * 2 * O.phase_N))
            out = R.run_diploid(O.phase_N, copies, 0, w, O.phase_gens, O.phase_reps, _seed(f"ph{p0:.2f}{q0:.2f}"))
            pf, pmale = out["p_f"], out["p_m"]
            m0 = (p0 + q0) / 2
            d1, d2 = op.iterate(op.autosomal_step, m0, m0, par, n_gen=3000)
            det_hi = bool(np.hypot(d1 - lo_eq["p1"], d2 - lo_eq["p2"]) > 0.05)
            ibm_hi = np.hypot(pf[:, -1] - lo_eq["p1"], pmale[:, -1] - lo_eq["p2"]) > 0.05
            rows.append(dict(p_start=m0, det_high=det_hi, ibm_frac_high=float(ibm_hi.mean()), n=O.phase_reps))
            trajs.append((pf[:4], pmale[:4]))
    df = pd.DataFrame(rows).groupby("p_start", as_index=False).first()
    io.save_table(df, "owen_bistability_ibm")
    agree = float(np.mean([(r.ibm_frac_high > 0.5) == r.det_high for r in df.itertuples()]))

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.contourf(gb, gb, basin_hi.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#e7eef9", "#fdf0d8"])
    speed = np.hypot(U, Vv)
    ax.streamplot(g, g, U, Vv, color=np.log10(speed + 1e-9), cmap=V.SEQ, density=1.3, linewidth=0.9, arrowsize=0.9)
    for pf, pmale in trajs[:: max(1, len(trajs) // 25)]:
        for i in range(pf.shape[0]):
            ax.plot(pf[i], pmale[i], color=V.INK2, lw=0.5, alpha=0.5)
    for e in eq:
        ax.plot(e["p1"], e["p2"], marker="o" if e["stable"] else "X", ms=11,
                color=V.CAT[3] if e["stable"] else V.CAT[7], mec=V.INK, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="p1 (freq. A, females)", ylabel="p2 (freq. A, males)",
           title="Owen (1953) bistability: flow of the recursion, basins (shading),\n"
                 "stable equilibria (o), saddle (X), IBM trajectories (grey)")
    ax.grid(False)
    V.save(fig, io.fig_path("O4_owen_phase_portrait"))

    # animation of an IBM cloud crossing the plane
    N = O.phase_N
    starts2 = [(p, p) for p in np.linspace(0.15, 0.85, 15)]
    clouds = []
    for i, (p0, _) in enumerate(starts2):
        out = R.run_diploid(N, int(round(p0 * 2 * N)), 0, w, O.phase_gens, 6, _seed(f"anim{i}"))
        clouds.append(np.stack([out["p_f"], out["p_m"]], -1))
    clouds = np.concatenate(clouds, 0)  # (pops, t, 2)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.contourf(gb, gb, basin_hi.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#e7eef9", "#fdf0d8"])
    ax.streamplot(g, g, U, Vv, color=V.AXIS, density=1.0, linewidth=0.6, arrowsize=0.7)
    for e in eq:
        ax.plot(e["p1"], e["p2"], marker="o" if e["stable"] else "X", ms=11,
                color=V.CAT[3] if e["stable"] else V.CAT[7], mec=V.INK, zorder=5)
    sc = ax.scatter(clouds[:, 1, 0], clouds[:, 1, 1], s=26, color=V.CAT[0], edgecolor=V.SURFACE, lw=1, zorder=6)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", color=V.INK)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="p1 (females)", ylabel="p2 (males)",
           title=f"IBM populations (N={N}) settling into Owen's two attractors")
    ax.grid(False)

    def upd(t):
        sc.set_offsets(clouds[:, max(t, 1), :])
        txt.set_text(f"generation {t}")
        return sc, txt

    frames = list(range(1, O.phase_gens + 1, max(1, O.phase_gens // 120)))
    FuncAnimation(fig, upd, frames=frames).save(io.anim_path("owen_bistability"), writer=PillowWriter(fps=15))
    plt.close(fig)
    return dict(equilibria=[dict(p1=e["p1"], p2=e["p2"], stable=e["stable"],
                                 eig=[complex(z).real for z in e["eig"]]) for e in eq],
                basin_agreement=agree,
                blueprint_cubic=[tuple(map(float, c)) for c in op.blueprint_cubic_equilibria(*par)])


ALL = [("O1 autosomal invasion", autosomal_invasion), ("O2 Parsons example", worked_example),
       ("O3 X-linked invasion", xlinked_invasion), ("O4 Owen bistability", owen_bistability)]

"""Mother's curse (Havird et al. 2019 review) formalised and tested:
  C1 strong form: a female-beneficial mito variant spreads whatever it does to males;
  C2 'twofold' rule: a male-sterilising *nuclear* allele needs >= 2x female fitness,
     a mito one needs only a slight female benefit (Parsons' latent root);
  C3 weak form: female-neutral, male-harming mito variants drift (P_fix = x0);
  C4 nuclear restorer of male fitness invades behind the mito sweep."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as C
from .. import io
from ..analysis import stats as S
from ..deterministic import mothers_curse as mc
from ..ibm import runners as R
from ..viz import style as V

Q = C.CURSE


def _seed(name):
    return C.seed_for("curse/" + name)


def strong_form():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    t = np.arange(Q.gens + 1)
    for i, s_m in enumerate(Q.s_m_values):
        W = mc.fitness_tables(s_f=Q.s_f, s_m=s_m)
        Z = mc.iterate(mc.initial(Q.x0, 0.0), W, Q.gens)
        x_det = mc.mito_freq(Z)
        out = R.run_diploid(Q.N, 0, int(Q.x0 * Q.N), C.cytonuclear_viability(W), Q.gens, Q.reps, _seed(f"s{s_m}"))
        x = out["mito"]
        axs[0].plot(t, x.mean(0), color=V.CAT[i], lw=1.6, label=f"IBM mean, male cost s_m={s_m}")
        rows.append(dict(s_m=s_m, det_final=float(x_det[-1]), ibm_mean_final=float(x[:, -1].mean()),
                         ibm_fixed=float(np.mean(x[:, -1] == 1)), ibm_lost=float(np.mean(x[:, -1] == 0)),
                         rmse=S.rmse(x.mean(0), x_det)))
    axs[0].plot(t, x_det, color=V.INK, lw=2, ls="--", label="deterministic (identical for every s_m)")
    axs[0].set(xlabel="generation", ylabel="freq. of male-harming mito variant",
               title=f"Strong mother's curse: female benefit s_f={Q.s_f}")
    axs[0].legend(fontsize=7.5)
    df = pd.DataFrame(rows)
    io.save_table(df, "curse_strong")
    ks = S.ks_two_sample  # endpoint distributions should not differ between s_m values
    finals = {}
    for s_m in Q.s_m_values:
        W = mc.fitness_tables(s_f=Q.s_f, s_m=s_m)
        out = R.run_diploid(Q.N, 0, int(Q.x0 * Q.N), C.cytonuclear_viability(W), 150, Q.reps, _seed(f"k{s_m}"))
        finals[s_m] = out["mito"][:, -1]
        axs[1].hist(finals[s_m], bins=20, histtype="step", lw=1.6, color=V.CAT[Q.s_m_values.index(s_m)],
                    label=f"s_m={s_m}")
    axs[1].set(xlabel="mito variant freq. at generation 150", ylabel="populations",
               title="Endpoint distributions overlap regardless of male cost")
    axs[1].legend(fontsize=8)
    V.save(fig, io.fig_path("C1_strong_mothers_curse"))
    kst = {f"s_m=0 vs {s}": ks(finals[0.0], finals[s]) for s in Q.s_m_values[1:]}
    return dict(table=df.to_dict("records"), ks_endpoint=kst)


def twofold():
    """Male-sterilising variant (male carriers have fitness 0): maternal
    (mito) vs biparental (dominant nuclear) inheritance. Verdict: does the
    ensemble-mean frequency at the horizon rise above the founding dose
    (95% CI), compared with the oracle trajectory from the same dose."""
    from ..deterministic import owen_parsons as op
    rows = []
    f0, T = Q.twofold_x0, Q.twofold_gens

    def direction(final, start, det_ratio):
        ratio = final.mean() / start
        se = final.std(ddof=1) / np.sqrt(len(final)) / start
        lo, hi = ratio - 1.96 * se, ratio + 1.96 * se
        v = "IBM: spreads" if lo > 1 else ("IBM: declines" if hi < 1 else "IBM: unresolved")
        agree = (v == "IBM: spreads" and det_ratio > 1) or (v == "IBM: declines" and det_ratio < 1)
        return dict(ibm_ratio=float(ratio), lo=float(lo), hi=float(hi), det_ratio=float(det_ratio),
                    verdict=v, agree=bool(agree))

    for s_f in Q.twofold_sf:
        W = mc.fitness_tables(s_f=s_f, s_m=1.0)
        out = R.run_diploid(Q.twofold_N, 0, int(f0 * Q.twofold_N), C.cytonuclear_viability(W), T,
                            Q.reps, _seed(f"tm{s_f}"))
        det_m = mc.mito_freq(mc.iterate(mc.initial(f0, 0.0), W, T)[-1]) / f0
        k_m = int(np.sum(np.nanmax(out["mito"], 1) > 5 * f0))
        rows.append(dict(kind="mitochondrial", female_fitness=1 + s_f, lam=float(mc.mito_invasion_rate(s_f)),
                         P_reach_5x=k_m / Q.reps, **direction(out["mito"][:, -1], f0, det_m)))
        w = C.owen_viability(1 + s_f, 1 + s_f, 1.0, 0.0, 0.0, 1.0)
        out = R.run_diploid(Q.twofold_N, int(f0 * 2 * Q.twofold_N), 0, w, T, Q.reps, _seed(f"tn{s_f}"))
        p1, p2 = op.iterate(op.autosomal_step, f0, f0, (1 + s_f, 1 + s_f, 1.0, 0.0, 0.0, 1.0), n_gen=T)
        k_n = int(np.sum(np.nanmax(out["p_zyg"], 1) > 5 * f0))
        rows.append(dict(kind="nuclear (dominant)", female_fitness=1 + s_f,
                         lam=float(mc.nuclear_invasion_rate(1 + s_f, 0.0)), P_reach_5x=k_n / Q.reps,
                         **direction(out["p_zyg"][:, -1], f0, float((p1 + p2) / 2) / f0)))
    df = pd.DataFrame(rows)
    df["det_invades"] = df.lam > 1
    io.save_table(df, "curse_twofold")

    # oracle curves on a fine grid
    grid = np.linspace(1.0, max(Q.twofold_sf) + 1, 120)[1:]
    det_mito = [mc.mito_freq(mc.iterate(mc.initial(f0, 0.0), mc.fitness_tables(s_f=g - 1, s_m=1.0), T)[-1]) / f0
                for g in grid]
    det_nuc = []
    for g in grid:
        p1, p2 = op.iterate(op.autosomal_step, f0, f0, (g, g, 1.0, 0.0, 0.0, 1.0), n_gen=T)
        det_nuc.append(float((p1 + p2) / 2) / f0)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for i, (kind, det) in enumerate((("mitochondrial", det_mito), ("nuclear (dominant)", det_nuc))):
        d = df[df.kind == kind]
        ax.plot(grid, det, color=V.CAT[i], lw=1.6, ls="--")
        ax.errorbar(d.female_fitness, d.ibm_ratio, yerr=V.err(d.ibm_ratio - d.lo, d.hi - d.ibm_ratio), fmt="o",
                    color=V.CAT[i], capsize=2, ms=6, label=f"IBM: {kind} variant (mean ± 95% CI)")
    ax.plot([], [], color=V.INK, ls="--", label="oracle")
    ax.axhline(1, color=V.AXIS, lw=1)
    for x, txt in ((1.0, "mito threshold:\nany benefit"), (2.0, "nuclear threshold:\ntwofold")):
        ax.axvline(x, color=V.MUTED, lw=1, ls=":")
        ax.text(x, 70, txt, color=V.INK2, fontsize=8, ha="center", va="bottom")
    ax.set(yscale="log", ylim=(0.02, 150), xlabel="female fitness of carriers (relative to 1); male carriers sterile",
           ylabel=f"frequency at t={T} / founding frequency",
           title="A male-sterilising variant: maternal vs biparental inheritance\n")
    ax.legend(fontsize=8, loc="lower right")
    V.save(fig, io.fig_path("C2_twofold_rule"))
    decided = df[df.verdict != "IBM: unresolved"]
    return dict(verdicts=df.verdict.value_counts().to_dict(), agreement_among_decided=float(decided.agree.mean()),
                table=df.to_dict("records"))


def weak_form():
    rows = []
    for s_m in (0.0, 0.5, 1.0):
        W = mc.fitness_tables(s_f=0.0, s_m=s_m)
        out = R.run_diploid(Q.weak_N, 0, int(Q.weak_x0 * Q.weak_N), C.cytonuclear_viability(W), Q.weak_gens,
                            Q.weak_reps, _seed(f"w{s_m}"))
        fin = out["mito"][:, -1]
        # >= 0.95 counts as fixed: with s_m = 1 the last wild-type males can
        # vanish first, collapsing the population just short of fixation
        k = int(np.sum(fin >= 0.95))
        p_, l_, h_ = S.wilson(k, Q.weak_reps)
        rows.append(dict(s_m=s_m, P_fix=float(p_), lo=float(l_), hi=float(h_), unresolved=float(np.mean((fin > 0) & (fin < 1))),
                         det_prediction_neutral=Q.weak_x0))
    df = pd.DataFrame(rows)
    io.save_table(df, "curse_weak")
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.bar([str(s) for s in df.s_m], df.P_fix, color=V.CAT[0], width=0.5)
    ax.errorbar(range(len(df)), df.P_fix, yerr=V.err(df.P_fix - df.lo, df.hi - df.P_fix), fmt="none", ecolor=V.INK)
    ax.axhline(Q.weak_x0, color=V.INK, ls="--", lw=1.4, label=f"neutral expectation = initial freq. {Q.weak_x0}")
    ax.set(xlabel="male cost s_m (female effect = 0)", ylabel="P(fixation)",
           title=f"Weak mother's curse: drift alone (N={Q.weak_N})")
    ax.legend(fontsize=8)
    V.save(fig, io.fig_path("C3_weak_mothers_curse"))
    return dict(table=df.to_dict("records"))


def restorer():
    """Two scenarios, deterministic vs IBM:
      simultaneous: mito variant (5%) and costly restorer (2%) appear together;
      sequential  : restorer (2%) appears once the mito variant is common (95%).
    Control: restorer without any mito variant (pure cost -> purged)."""
    rp = Q.restorer
    W = mc.fitness_tables(**rp)
    T = Q.restorer_gens
    t = np.arange(T + 1)
    scen = {"simultaneous": (Q.restorer_x0, Q.restorer_p0), "sequential": (0.95, Q.restorer_p0),
            "control (no mito variant)": (0.0, Q.restorer_p0)}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    res = {}
    for ax, (name, (x0, p0)) in zip(axs, scen.items()):
        Z = mc.iterate(mc.initial(x0, p0), W, T)
        x_det, p_det = mc.mito_freq(Z), mc.nuclear_freq(Z)
        out = R.run_diploid(Q.N, int(p0 * 2 * Q.N), int(x0 * Q.N), C.cytonuclear_viability(W), T, Q.reps,
                            _seed("restorer" + name))
        for arr, det, c, lab in ((out["mito"], x_det, V.CAT[0], "mito variant"),
                                 (out["p_zyg"], p_det, V.CAT[1], "nuclear restorer")):
            # outcomes are bimodal (lost / fixed): show individual populations
            for traj in arr[:20]:
                ax.plot(t, traj, color=c, lw=0.5, alpha=0.35)
            ax.plot(t, arr.mean(0), color=c, lw=2, label=f"IBM mean: {lab}")
            ax.plot(t, det, color=c, lw=1.6, ls="--")
        ax.plot([], [], color=V.INK, ls="--", label="deterministic")
        ax.set(xlabel="generation", title=f"{name}: start mito={x0}, restorer={p0}")
        res[name] = dict(det_final=dict(mito=float(x_det[-1]), restorer=float(p_det[-1])),
                         ibm_mean_final=dict(mito=float(out["mito"][:, -1].mean()),
                                             restorer=float(out["p_zyg"][:, -1].mean())),
                         ibm_P_restorer_lost=float(np.mean(out["p_zyg"][:, -1] == 0)),
                         ibm_P_restorer_fixed=float(np.mean(out["p_zyg"][:, -1] == 1)))
    axs[0].set_ylabel("frequency")
    axs[0].legend(fontsize=7.5)
    fig.suptitle(f"Arms race: mito variant (s_f={rp['s_f']}, s_m={rp['s_m']}) and a restorer of male fitness "
                 f"(cost {rp['cost_R']}, h={rp['h_R']}), N={Q.N}; thin lines = 20 individual IBM populations",
                 color=V.INK)
    V.save(fig, io.fig_path("C4_restorer"))
    return res


ALL = [("C1 strong form", strong_form), ("C2 twofold rule", twofold), ("C3 weak form", weak_form),
       ("C4 restorer", restorer)]

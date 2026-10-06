"""Extended test suite: classical sexually-antagonistic theory, drift theory,
and ML analyses -- each confronted with the zero-leakage IBM.

E1  Kidwell/Fry autosomal protected-polymorphism wedges for three dominance
    scenarios (+ M1 GP denoiser extracting the IBM's emergent region)
E2  X-linked vs autosomal (Rice 1984 vs Fry 2010)
E3  Kimura fixation probabilities: nuclear (sex-averaged s) and maternal mito
M2  neural emulator trained on IBM fixation data, tested against Kimura
E4  SA locus linked to the sex-determining region (Rice 1987)
E5  diffusion (backward Kolmogorov) time to loss of the IaSC polymorphism
M3  inverse problem: recover (s_f, s_m) from IBM trajectories
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .. import config as C
from .. import io
from ..analysis import ml
from ..analysis import stats as S
from ..deterministic import owen_parsons as op
from ..deterministic import poster_model as pm
from ..deterministic import sa_classics as sa
from ..ibm import runners as R
from ..viz import style as V

X = C.EXT
log = io.log


def _seed(name):
    return C.seed_for("ext/" + name)


def _ratio(final, start):
    final = np.asarray(final, float)
    final = final[np.isfinite(final)]
    r = final.mean() / start
    se = final.std(ddof=1) / np.sqrt(len(final)) / start
    return r, se


# =========================================================================
# E1 + E2 (+ M1): protected polymorphism maps
# =========================================================================
def _map(kind, h_f, h_m, grid_n, reps, name):
    s = np.linspace(X.s_lo, X.s_hi, grid_n)
    rows = []
    for s_f in s:
        for s_m in s:
            rec = dict(s_f=s_f, s_m=s_m)
            for allele in ("A1", "A2"):
                if kind == "autosomal":
                    w = C.kidwell_autosomal_viability(s_f, s_m, h_f, h_m, counted=allele)
                    copies = int(round(X.dose * 2 * X.N_map))
                    out = R.run_diploid(X.N_map, copies, 0, w, X.T_map, reps, _seed(f"{name}{allele}{s_f:.3f}{s_m:.3f}"))
                    final, start = out["p_zyg"][:, -1], copies / (2 * X.N_map)
                    par = sa.owen_params(s_f, s_m, h_f, h_m)
                    if allele == "A2":  # oracle with the same relabelling
                        par = (par[2], par[1], par[0], par[5], par[4], par[3])
                    d1, d2 = op.iterate(op.autosomal_step, start, start, par, n_gen=X.T_map)
                    det = float((d1 + d2) / 2) / start
                else:
                    w = C.kidwell_xlinked_viability(s_f, s_m, h_f, counted=allele)
                    copies = int(round(X.dose * 1.5 * X.N_map))
                    out = R.run_xlinked(X.N_map, copies, w, X.T_map, reps, _seed(f"{name}{allele}{s_f:.3f}{s_m:.3f}"))
                    pall = (2 * out["p_f"] + out["p_m"]) / 3
                    final, start = pall[:, -1], np.nanmean(pall[:, 0])
                    a1, h1, b1, a2, b2 = sa.xlinked_params(s_f, s_m, h_f)
                    par = (a1, h1, b1, a2, b2) if allele == "A1" else (b1, h1, a1, b2, a2)
                    d1, d2 = op.iterate(op.sexlinked_step, start, start, par, n_gen=X.T_map)
                    det = float((2 * d1 + d2) / 3) / start
                r, se = _ratio(final, start)
                rec[f"{allele}_ibm_ratio"], rec[f"{allele}_ibm_se"], rec[f"{allele}_det_ratio"] = r, se, det
            if kind == "autosomal":
                rec["oracle_protected"] = bool(sa.fry_eq3(s_f, s_m, h_f, h_m))
            else:
                rec["oracle_protected"] = bool(sa.rice_eq2(s_f, s_m, h_f))
            rows.append(rec)
    df = pd.DataFrame(rows)
    for a in ("A1", "A2"):
        df[f"{a}_ibm_up"] = (df[f"{a}_ibm_ratio"] - 1.96 * df[f"{a}_ibm_se"]) > 1
        df[f"{a}_ibm_down"] = (df[f"{a}_ibm_ratio"] + 1.96 * df[f"{a}_ibm_se"]) < 1
    df["ibm_protected"] = df.A1_ibm_up & df.A2_ibm_up
    df["ibm_not_protected"] = df.A1_ibm_down | df.A2_ibm_down
    return df


def _gp_region(df, fine):
    """M1: GP-denoise log(ratio) for both invasion directions using only IBM
    data; region where both predicted log-ratios > 0."""
    Xg = df[["s_f", "s_m"]].values
    out = {}
    for a in ("A1", "A2"):
        # tanh(log r / 2) keeps the sign of log r (the boundary is at 0) but
        # compresses the huge values far from the boundary that otherwise
        # dominate a stationary GP; variance by the delta method
        r = df[f"{a}_ibm_ratio"].clip(lower=1e-3).values
        y = np.tanh(np.log(r) / 2)
        var = ((1 - y ** 2) / (2 * r) * df[f"{a}_ibm_se"].values) ** 2
        m, sd, _ = ml.gp_denoise(Xg, y, var, fine, length_scale=0.3, seed=1)
        out[a] = m
    return (out["A1"] > 0) & (out["A2"] > 0), out


def polymorphism_maps():
    fine_s = np.linspace(X.s_lo, X.s_hi, 80)
    FF, MM = np.meshgrid(fine_s, fine_s)
    fine = np.column_stack([FF.ravel(), MM.ravel()])
    results, tables = {}, {}
    fig, axs = plt.subplots(2, len(X.scenarios), figsize=(5.2 * len(X.scenarios), 9.4), sharex=True, sharey=True)
    for col, (label, h_f, h_m) in enumerate(X.scenarios):
        for row, kind in enumerate(("autosomal", "X-linked")):
            t0 = time.time()
            grid_n = X.grid_auto if kind == "autosomal" else X.grid_x
            reps = X.reps_map if kind == "autosomal" else X.reps_map_x
            df = _map(kind, h_f, h_m, grid_n, reps, f"{kind}{label}")
            tables[(kind, label)] = df.assign(kind=kind, scenario=label)
            oracle_fine = (sa.fry_eq3(FF, MM, h_f, h_m) if kind == "autosomal" else sa.rice_eq2(FF, MM, h_f)).ravel()
            gp_fine, _ = _gp_region(df, fine)
            # raw IBM verdicts upsampled by nearest grid cell
            idx = np.argmin((fine[:, None, 0] - df.s_f.values[None]) ** 2 + (fine[:, None, 1] - df.s_m.values[None]) ** 2, 1)
            raw_fine = df.ibm_protected.values[idx]
            decided = df[df.ibm_protected | df.ibm_not_protected]
            res = dict(oracle_area=float(oracle_fine.mean()), gp_area=float(gp_fine.mean()),
                       iou_gp=ml.iou(gp_fine, oracle_fine), iou_raw=ml.iou(raw_fine, oracle_fine),
                       cell_agreement=float((decided.ibm_protected == decided.oracle_protected).mean()),
                       n_cells=len(df), n_decided=len(decided), seconds=round(time.time() - t0, 1))
            results[f"{kind} | {label}"] = res
            log.info("  E1/E2 %s | %s: %s", kind, label, res)
            ax = axs[row, col]
            ax.contourf(fine_s, fine_s, oracle_fine.reshape(FF.shape).astype(float), levels=[0.5, 1.5],
                        colors=["#fbeccb"])
            ax.contour(fine_s, fine_s, oracle_fine.reshape(FF.shape).astype(float), levels=[0.5], colors=[V.INK],
                       linewidths=1.8)
            ax.contour(fine_s, fine_s, gp_fine.reshape(FF.shape).astype(float), levels=[0.5], colors=[V.CAT[0]],
                       linewidths=1.6, linestyles="--")
            c = np.where(df.ibm_protected, V.CAT[1], np.where(df.ibm_not_protected, V.CAT[0], V.MUTED))
            ax.scatter(df.s_f, df.s_m, c=c, s=26, edgecolor=V.SURFACE, linewidth=0.8, zorder=3)
            ax.set(title=f"{kind}: {label}", xlabel="s_f" if row == 1 else None, ylabel="s_m" if col == 0 else None)
            ax.text(0.03, 0.97, f"oracle area {res['oracle_area']:.2f}\nGP-IBM area {res['gp_area']:.2f}\n"
                                f"IoU {res['iou_gp']:.2f}", transform=ax.transAxes, va="top", fontsize=8,
                    color=V.INK2, bbox=dict(facecolor=V.SURFACE, edgecolor=V.GRID, pad=3))
            ax.grid(False)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=V.INK, lw=1.8), Line2D([], [], color=V.CAT[0], lw=1.6, ls="--"),
                Line2D([], [], marker="o", color=V.CAT[1], lw=0), Line2D([], [], marker="o", color=V.CAT[0], lw=0),
                Line2D([], [], marker="o", color=V.MUTED, lw=0)],
               ["oracle: protected polymorphism (Fry eq. 3 / Rice eq. 2)", "M1: GP-denoised IBM boundary",
                "IBM: both alleles invade", "IBM: one allele cannot invade", "IBM: undecided"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.suptitle("Where can sexually antagonistic polymorphism be maintained? Autosomes vs X, three dominance "
                 "scenarios", y=1.06, color=V.INK)
    V.save(fig, io.fig_path("E1_E2_polymorphism_maps"))
    io.save_table(pd.concat(tables.values(), ignore_index=True), "ext_polymorphism_maps")
    return results


# =========================================================================
# E3: Kimura fixation probabilities
# =========================================================================
def _pfix(final):
    final = np.asarray(final, float)
    return float(np.mean(final == 1) + np.mean(np.where((final > 0) & (final < 1), final, 0))), \
        float(np.mean((final > 0) & (final < 1)))


def kimura():
    rows = []
    for N in X.kim_N:
        gens = 6 * N
        for S_ in X.kim_S:
            s = S_ / (4 * N)
            copies = int(round(X.kim_p0 * 2 * N))
            for scen, (s_f, s_m) in (("concordant", (s, s)),
                                     ("sexually antagonistic", (s + X.kim_sa_spread, s - X.kim_sa_spread))):
                out = R.run_diploid(N, copies, 0, C.genic_viability(s_f, s_m), gens, X.kim_reps,
                                    _seed(f"kim{N}{S_}{scen}"))
                pf, unabs = _pfix(out["p_zyg"][:, -1])
                par = tuple(np.r_[C.genic_viability(s_f, s_m)[0, [2, 1, 0]], C.genic_viability(s_f, s_m)[1, [2, 1, 0]]])
                Mfun = lambda p, par=par: 0.5 * np.add(*op.autosomal_step(p, p, *par)) - p
                full = float(sa.fixation_probability(Mfun, lambda p, N=N: p * (1 - p) / (2 * N), X.kim_p0))
                rows.append(dict(kind=scen, N=N, S=S_, s_f=s_f, s_m=s_m, p_fix=pf, unabsorbed=unabs,
                                 kimura=float(sa.kimura_u(X.kim_p0, N, (s_f + s_m) / 2)),
                                 diffusion_full=full, deterministic=float((s_f + s_m) > 0)))
            # maternal mito: female selection only, Ne = N_f = N/2
            Wm = np.array([[1.0, 1, 1, 1 + 2 * s, 1 + 2 * s, 1 + 2 * s], [1.0, 1, 1, 0.5, 0.5, 0.5]])
            out = R.run_diploid(N, 0, int(round(X.kim_p0 * N)), Wm, gens, X.kim_reps, _seed(f"kimm{N}{S_}"))
            pf, unabs = _pfix(out["mito"][:, -1])
            ku = float(sa.kimura_u_haploid(X.kim_p0, N / 2, 2 * s))
            rows.append(dict(kind="maternal mito (male cost 0.5)", N=N, S=S_, s_f=2 * s, s_m=-0.5, p_fix=pf,
                             unabsorbed=unabs, kimura=ku, diffusion_full=ku, deterministic=float(s > 0)))
        log.info("  E3 N=%d done", N)
    df = pd.DataFrame(rows)
    df["se"] = np.sqrt(df.p_fix * (1 - df.p_fix) / X.kim_reps)
    df["z"] = (df.p_fix - df.kimura) / np.maximum(df.se, 1e-3)
    df["z_full"] = (df.p_fix - df.diffusion_full) / np.maximum(df.se, 1e-3)
    io.save_table(df, "ext_kimura")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, kind in zip(axs, df.kind.unique()):
        d0 = df[df.kind == kind]
        for i, N in enumerate(X.kim_N):
            d = d0[d0.N == N]
            ax.errorbar(d.S, d.p_fix, yerr=1.96 * d.se, fmt="o", color=V.CAT[i], ms=6, capsize=2, label=f"IBM N={N}")
        Sg = np.linspace(min(X.kim_S), max(X.kim_S), 200)
        N0 = X.kim_N[-1]
        if "mito" in kind:
            ax.plot(Sg, sa.kimura_u_haploid(X.kim_p0, N0 / 2, 2 * Sg / (4 * N0)), color=V.INK, lw=1.8,
                    label="Kimura, haploid, Ne = N/2")
            ax.set_xlabel("S = 4N s (female selection on the mito variant: s_f = 2s)")
        else:
            ax.plot(Sg, sa.kimura_u(X.kim_p0, N0, Sg / (4 * N0)), color=V.INK, lw=1.8,
                    label="Kimura with s = (s_f + s_m)/2")
            ax.set_xlabel("S = 4N(s_f + s_m)/2")
        if "antag" in kind:
            for i, N in enumerate(X.kim_N):
                d = d0[d0.N == N]
                ax.plot(d.S, d.diffusion_full, marker="x", ls="", color=V.CAT[i], ms=8, mew=2,
                        label=f"full diffusion (sex-specific M), N={N}")
        ax.step(Sg, (Sg > 0).astype(float), where="mid", color=V.MUTED, ls=":", lw=1.4,
                label="deterministic (infinite N)")
        ax.set(title=kind + (f" (s_f-s_m = {2 * X.kim_sa_spread})" if "antag" in kind else ""), ylim=(-0.03, 1.03))
        ax.legend(fontsize=7.5, loc="upper left")
    axs[0].set_ylabel(f"P(fixation) from p0 = {X.kim_p0}")
    fig.suptitle("Fixation probabilities: IBM vs Kimura's diffusion formula (proven result) vs the deterministic step",
                 color=V.INK)
    V.save(fig, io.fig_path("E3_kimura"))
    by = df.groupby("kind").apply(lambda d: pd.Series(dict(
        rmse=S.rmse(d.p_fix, d.kimura), max_abs_z=float(d.z.abs().max()),
        frac_within_2se=float((d.z.abs() < 2).mean()),
        rmse_full_diffusion=S.rmse(d.p_fix, d.diffusion_full),
        frac_within_2se_full=float((d.z_full.abs() < 2).mean()),
        rmse_deterministic=S.rmse(d.p_fix, d.deterministic))), include_groups=False)
    return by.to_dict("index")


# =========================================================================
# M2: neural emulator of the IBM vs Kimura (held-out N)
# =========================================================================
def emulator():
    rng = np.random.default_rng(_seed("emu"))
    rows = []
    for i in range(X.emu_n):
        N = int(rng.choice([150, 250, 400, 600, 800]))
        S_ = rng.uniform(-3, 9)
        s = S_ / (4 * N)
        out = R.run_diploid(N, int(round(X.kim_p0 * 2 * N)), 0, C.genic_viability(s, s), 6 * N, X.emu_reps,
                            _seed(f"emu{i}"))
        pf, _ = _pfix(out["p_zyg"][:, -1])
        rows.append(dict(N=N, s=s, S=S_, p_fix=pf))
    df = pd.DataFrame(rows)
    io.save_table(df, "ext_emulator_training")
    test = df.N == 800                        # extrapolation target: largest N never seen
    tr = ~test
    y = ml.logit(df.p_fix)
    res, preds = {}, {}
    for name, feats in (("generic inputs (log N, s)", np.column_stack([np.log(df.N), df.s * 100])),
                        ("diffusion-scaled input (N*s)", np.column_stack([df.N * df.s]))):
        model = ml.mlp(hidden=(32, 32), seed=0)
        model.fit(feats[tr], y[tr])
        p = ml.expit(model.predict(feats))
        kim = sa.kimura_u(X.kim_p0, df.N, df.s)
        res[name] = dict(rmse_vs_ibm_test=S.rmse(p[test], df.p_fix[test]), rmse_vs_kimura_test=S.rmse(p[test], kim[test]),
                         rmse_vs_kimura_train=S.rmse(p[tr], kim[tr]))
        preds[name] = p
    res["ibm_vs_kimura_test"] = S.rmse(df.p_fix[test], sa.kimura_u(X.kim_p0, df.N, df.s)[test])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    Sg = np.linspace(-3, 9, 200)
    ax.plot(Sg, sa.kimura_u(X.kim_p0, 800, Sg / 3200), color=V.INK, lw=2, label="Kimura (proven), N = 800")
    d = df[test]
    ax.scatter(d.S, d.p_fix, color=V.MUTED, s=18, label="IBM, N = 800 (held out)")
    for i, (name, p) in enumerate(preds.items()):
        o = np.argsort(df.S[test].values)
        ax.plot(df.S[test].values[o], p[test.values][o], color=V.CAT[i], lw=1.6, label=f"MLP: {name}")
    ax.set(xlabel="S = 4Ns", ylabel="P(fixation)", title="M2: an emulator trained only on IBM data (N <= 600)")
    ax.legend(fontsize=7.5)
    V.save(fig, io.fig_path("M2_emulator"))
    return res


# =========================================================================
# E4: SA locus linked to the sex-determining region
# =========================================================================
def xy_linkage():
    h = 0.5
    rows = []
    for r in X.xy_r:
        for s_m in X.xy_sm:
            poly, pXe, pY = sa.xy_polymorphic(X.xy_sf, s_m, h, h, r)
            w = C.xy_viability(X.xy_sf, s_m, h, h)
            out = R.run_xy(X.xy_N, 0.5, 0.5, w, r, X.xy_T, X.xy_reps, _seed(f"xy{r}{s_m}"))
            pX, pYi = out["pX"][:, -1], out["pY"][:, -1]
            tot = (2 * pX + pYi) / 3
            rows.append(dict(r=r, s_m=s_m, oracle_poly=bool(poly), oracle_pX=float(pXe), oracle_pY=float(pY),
                             ibm_poly=float(np.mean((tot > 0) & (tot < 1))), ibm_pX=float(np.nanmean(pX)),
                             ibm_pY=float(np.nanmean(pYi)), autosomal_protected=bool(sa.fry_eq3(X.xy_sf, s_m, h, h))))
    df = pd.DataFrame(rows)
    io.save_table(df, "ext_xy_linkage")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax = axs[0]
    piv_o = df.pivot(index="s_m", columns="r", values="oracle_poly").astype(float)
    piv_i = df.pivot(index="s_m", columns="r", values="ibm_poly")
    xs = np.arange(len(X.xy_r))
    im = ax.imshow(piv_i.values, origin="lower", aspect="auto", cmap=V.SEQ, vmin=0, vmax=1,
                   extent=(-0.5, len(xs) - 0.5, -0.5, len(X.xy_sm) - 0.5))
    for i in range(piv_o.shape[0]):
        for j in range(piv_o.shape[1]):
            if piv_o.values[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=V.CAT[1], lw=2))
    ax.set_xticks(xs, [str(v) for v in X.xy_r])
    ax.set_yticks(range(len(X.xy_sm)), [f"{v:.2f}" for v in X.xy_sm])
    ax.set(xlabel="recombination with the sex-determining region, r", ylabel=f"s_m  (s_f = {X.xy_sf})",
           title="IBM: fraction polymorphic (colour); oracle polymorphic (orange box)")
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    ax = axs[1]
    d = df[np.isclose(df.s_m, min(X.xy_sm, key=lambda v: abs(v - X.xy_sf)))]
    xr = np.arange(len(d))
    ax.plot(xr, d.oracle_pY - d.oracle_pX, "o-", color=V.INK, label="oracle: pY - pX (male-benefit allele)")
    ax.plot(xr, d.ibm_pY - d.ibm_pX, "s--", color=V.CAT[0], label="IBM")
    ax.set_xticks(xr, [f"{v:g}" for v in d.r])
    ax.set(xlabel="recombination with the sex-determining region, r", ylabel="Y - X difference in A1 frequency",
           title="Rice (1987): tight linkage puts the male-benefit allele on the Y")
    ax.legend(fontsize=8)
    V.save(fig, io.fig_path("E4_xy_linkage"))
    return dict(agreement=float(np.mean((df.ibm_poly > 0.5) == df.oracle_poly)),
                oracle_poly_fraction_by_r=df.groupby("r").oracle_poly.mean().to_dict(),
                ibm_poly_fraction_by_r=df.groupby("r").ibm_poly.mean().to_dict(),
                ld_rmse=S.rmse(df.ibm_pY - df.ibm_pX, df.oracle_pY - df.oracle_pX))


# =========================================================================
# E5: diffusion time to loss of the IaSC polymorphism (poster, panel A)
# =========================================================================
def diffusion_persistence():
    b = C.POSTER.b_strong
    rows = []
    for a in X.dif_a:
        for N in X.dif_N:
            T_pred = sa.mean_absorption_time(lambda p: pm.one_locus_step(p, a, b) - p, lambda p: p * (1 - p) / N, 0.5)
            cap = int(max(2000, 8 * T_pred))
            w = C.poster_viability(a, b, 0, 1)
            out = R.run_haploid([0, N // 2, 0, N - N // 2], w, 0.5, cap, X.dif_reps, _seed(f"dif{a}{N}"))
            pA = out["pA1"]
            absorbed = (pA <= 0) | (pA >= 1)
            hit = absorbed.any(1)
            t = np.where(hit, absorbed.argmax(1), cap).astype(float)
            # tuner: effective size of the IBM measured from its own one-step
            # fluctuations around the deterministic mean change
            p0_, p1_ = pA[:, :-1].ravel(), pA[:, 1:].ravel()
            ok = (p0_ > 0) & (p0_ < 1)
            resid = p1_[ok] - pm.one_locus_step(p0_[ok], a, b)
            Ne_hat = float(np.mean(p0_[ok] * (1 - p0_[ok])) / np.mean(resid ** 2))
            T_cal = sa.mean_absorption_time(lambda p: pm.one_locus_step(p, a, b) - p,
                                            lambda p: p * (1 - p) / Ne_hat, 0.5)
            rows.append(dict(a=a, N=N, Ne_hat=Ne_hat, T_diffusion_calibrated=float(T_cal),
                             T_diffusion=float(T_pred), T_ibm=float(t.mean()),
                             T_ibm_se=float(t.std(ddof=1) / np.sqrt(len(t))), censored=float(1 - hit.mean()),
                             T_neutral=float(-2 * N * np.log(0.5))))
    df = pd.DataFrame(rows)
    df["rel_err"] = df.T_ibm / df.T_diffusion - 1
    df["rel_err_calibrated"] = df.T_ibm / df.T_diffusion_calibrated - 1
    io.save_table(df, "ext_diffusion")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, a in enumerate(X.dif_a):
        d = df[df.a == a]
        Ng = np.geomspace(min(X.dif_N), max(X.dif_N), 40)
        Tg = [sa.mean_absorption_time(lambda p: pm.one_locus_step(p, a, b) - p, lambda p, n=n: p * (1 - p) / n, 0.5)
              for n in Ng]
        ax.plot(Ng, Tg, color=V.CAT[i], lw=1.6)
        ax.errorbar(d.N, d.T_ibm, yerr=1.96 * d.T_ibm_se, fmt="o", color=V.CAT[i], ms=6, capsize=2, label=f"a = {a}")
        ax.plot(d.N, d.T_diffusion_calibrated, marker="_", ls="", color=V.CAT[i], ms=16, mew=2)
    ax.plot(Ng, 2 * Ng * np.log(2), color=V.MUTED, ls=":", label="neutral (2N ln 2)")
    ax.set(xscale="log", yscale="log", xlabel="population size N",
           ylabel="mean generations until IaSC is lost (from p = 0.5)",
           title="E5: diffusion theory (lines) vs IBM (dots), b = 0.2")
    ax.plot([], [], color=V.INK, lw=1.6, label="diffusion theory, Ne = N")
    ax.plot([], [], marker="_", ls="", color=V.INK, ms=16, mew=2, label="diffusion with Ne measured from IBM (tuner)")
    ax.legend(fontsize=8)
    V.save(fig, io.fig_path("E5_diffusion_persistence"))
    return dict(median_abs_rel_err=float(df.rel_err.abs().median()), max_abs_rel_err=float(df.rel_err.abs().max()),
                median_abs_rel_err_calibrated=float(df.rel_err_calibrated.abs().median()),
                max_abs_rel_err_calibrated=float(df.rel_err_calibrated.abs().max()),
                Ne_over_N=(df.Ne_hat / df.N).describe()[["mean", "min", "max"]].to_dict(),
                table=df.to_dict("records"))


# =========================================================================
# M3: inverse problem -- recover (s_f, s_m) from trajectories
# =========================================================================
def _traj(N, s_f, s_m, reps, seed):
    w = C.kidwell_autosomal_viability(s_f, s_m, 0.5, 0.5)
    out = R.run_diploid(N, N, 0, w, X.inv_T, reps, seed)  # A1 starts at 0.5
    idx = np.arange(0, X.inv_T + 1, X.inv_every)
    return np.concatenate([out["p_f"][:, idx], out["p_m"][:, idx]], 1)


def _oracle_traj(s_f, s_m):
    par = sa.owen_params(s_f, s_m, 0.5, 0.5)
    tr = op.iterate(op.autosomal_step, 0.5, 0.5, par, n_gen=X.inv_T + 1, record=True)  # (T+2, 2)
    # IBM adults at generation g are selected zygotes made from generation g-1 gametes
    idx = np.arange(0, X.inv_T + 1, X.inv_every) + 1
    return np.concatenate([tr[idx, 0], tr[idx, 1]])


def inverse_problem():
    rng = np.random.default_rng(_seed("inv"))
    res, rows = {}, []
    for N in X.inv_N:
        th_tr = rng.uniform(0.02, 0.2, (X.inv_train, 2))
        th_te = rng.uniform(0.02, 0.2, (X.inv_test, 2))
        Xtr = np.vstack([_traj(N, *t, 1, _seed(f"invtr{N}{i}")) for i, t in enumerate(th_tr)])
        Xte = np.vstack([_traj(N, *t, 1, _seed(f"invte{N}{i}")) for i, t in enumerate(th_te)])
        Xtr, Xte = np.nan_to_num(Xtr, nan=0.5), np.nan_to_num(Xte, nan=0.5)
        model = ml.mlp(hidden=(64, 64), seed=0, alpha=1e-3)
        model.fit(Xtr, th_tr)
        pred_nn = model.predict(Xte)
        pred_ls = np.array([least_squares(lambda th: _oracle_traj(*th) - x, x0=[0.1, 0.1],
                                          bounds=([0.0, 0.0], [0.5, 0.5])).x for x in Xte])
        for est, P_ in (("MLP (amortised, trained on IBM)", pred_nn), ("oracle least-squares fit", pred_ls)):
            res[f"N={N} | {est}"] = dict(rmse_s_f=S.rmse(P_[:, 0], th_te[:, 0]), rmse_s_m=S.rmse(P_[:, 1], th_te[:, 1]),
                                         bias_s_f=float(np.mean(P_[:, 0] - th_te[:, 0])),
                                         bias_s_m=float(np.mean(P_[:, 1] - th_te[:, 1])),
                                         rmse_difference=S.rmse(P_[:, 0] - P_[:, 1], th_te[:, 0] - th_te[:, 1]))
            for t, p in zip(th_te, P_):
                rows.append(dict(N=N, estimator=est, s_f=t[0], s_m=t[1], s_f_hat=p[0], s_m_hat=p[1]))
        log.info("  M3 N=%d done", N)
    df = pd.DataFrame(rows)
    io.save_table(df, "ext_inverse")
    fig, axs = plt.subplots(1, len(X.inv_N), figsize=(6 * len(X.inv_N), 4.6), sharey=True)
    for ax, N in zip(np.atleast_1d(axs), X.inv_N):
        for i, est in enumerate(df.estimator.unique()):
            d = df[(df.N == N) & (df.estimator == est)]
            ax.scatter(d.s_f, d.s_f_hat, s=16, color=V.CAT[i], alpha=0.8, label=f"{est}: s_f")
            ax.scatter(d.s_m, d.s_m_hat, s=16, color=V.CAT[i], marker="x", alpha=0.8, label=f"{est}: s_m")
        ax.plot([0, 0.22], [0, 0.22], color=V.INK, ls="--", lw=1)
        ax.set(xlabel="true selection coefficient", title=f"M3: recovering s_f, s_m from one IBM trajectory, N={N}")
    np.atleast_1d(axs)[0].set_ylabel("estimate")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    V.save(fig, io.fig_path("M3_inverse_problem"))
    return res


ALL = [("E1/E2 polymorphism maps (+M1 GP)", polymorphism_maps), ("E3 Kimura fixation", kimura),
       ("M2 neural emulator", emulator), ("E4 XY linkage", xy_linkage),
       ("E5 diffusion persistence", diffusion_persistence), ("M3 inverse problem", inverse_problem)]

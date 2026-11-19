"""Quick reproduction check: recompute headline numbers from the code and
compare them with the outputs committed in this repository.

Nothing under results/ is written. Expected values are read from the committed
results/summary.json (the stored full-profile run); none are typed in here.

Checks (deterministic, well under a minute):
  P0  p* formula vs the iterated one-locus recursion (20,000 generations)
  P1  fraction of (a, r) space where the IaSC fixed point is stable, for the
      poster's seven representative modifiers at b = 0.2 and b = 0.02
  O2  Parsons' worked example: latent root, deterministic trajectory endpoint,
      interior equilibrium
  O4  Owen-type bistable parameter set: interior equilibria and their
      stability, cross-checked against the cubic in the earlier AI blueprint
  RPT results/REPORT.md regenerated from results/summary.json in a temporary
      directory and compared byte-for-byte with the committed file

Optional (--ibm, about a minute more): re-run the P0 individual-based
populations at N = 500 with the stored seeds and print the RMSE next to the
stored value. This is informational, not pass/fail: the stored run does not
record which IBM backend (numba CPU or OpenCL GPU) produced the poster suite,
and the two backends are statistically, not bit-for-bit, equivalent.

Optional (--p4, strict): check that every stored P4 cell started with
0 < M1 founder copies < N, then replay the stored P4 IBM at N = 500 on the CPU
kernels with the stored seeds and compare the number of resolved populations
in each of its 20 (a, dose) cells with results/data/poster_threshold_ibm.parquet
(--p4-all: all three stored sizes, N = 500, 2000 and 6000, i.e. all 60 cells).
P4 is the only caller of run_haploid(stop_on_m_absorb=True), so this checks
that the generation-0 forward-fill fix leaves the stored results unchanged.

usage:  python scripts/reproduce.py [--ibm] [--p4 | --p4-all]
exit status 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import filecmp
import importlib.util
import io as _io
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
os.environ["SC_PROFILE"] = "full"   # the committed results are the full profile

import numpy as np  # noqa: E402

from sexconflict import config as C  # noqa: E402
from sexconflict.deterministic import owen_parsons as op  # noqa: E402
from sexconflict.deterministic import poster_model as pm  # noqa: E402


class Checks:
    def __init__(self):
        self.rows = []

    def close(self, name, got, want, atol=0.0, rtol=1e-9):
        ok = want is not None and abs(float(got) - float(want)) <= atol + rtol * abs(float(want))
        self.rows.append((name, f"{float(got):.10g}", "missing" if want is None else f"{float(want):.10g}", ok))

    def equal(self, name, got, want):
        self.rows.append((name, str(got), str(want), got == want))

    def report(self):
        w = max(len(r[0]) for r in self.rows)
        print(f"{'check':<{w}}  {'recomputed':>18}  {'stored':>18}  status")
        for name, got, want, ok in self.rows:
            print(f"{name:<{w}}  {got:>18}  {want:>18}  {'PASS' if ok else 'FAIL'}")
        n_ok = sum(r[3] for r in self.rows)
        print(f"\n{n_ok}/{len(self.rows)} checks passed")
        return n_ok == len(self.rows)


def poster_checks(ck, S):
    P = C.POSTER
    b = P.b_strong
    lo, hi = pm.polymorphism_range(b)
    # P0, exactly as sexconflict/experiments/poster_exp.py::one_locus
    a_fine = np.linspace(lo - 0.01, hi + 0.01, 400)
    p_formula = np.clip(pm.pstar(a_fine, b), 0, 1)
    p = np.full_like(a_fine, 0.5)
    for _ in range(20000):
        p = pm.one_locus_step(p, a_fine, b)
    inner = (a_fine > lo + 1e-3) & (a_fine < hi - 1e-3)
    stored = S["poster"]["P0 one-locus polymorphism"]["iterate_vs_formula"]
    ck.close("P0 max |iterated - p*|", np.abs(p - p_formula)[inner].max(), stored, rtol=1e-6)
    # P1, exactly as poster_exp.py::stability_heatmaps ("representative")
    rep = S["poster"]["P1 stability heatmaps"]["representative"]
    half_cell = 0.5 / (P.ar_n * P.ar_n)      # identical count of stable grid cells
    for k in P.k_pairs:
        for bb, key in ((P.b_strong, "b02"), (P.b_weak, "b002")):
            got = float(pm.stable_fraction(bb, *k, n_a=P.ar_n, n_r=P.ar_n))
            ck.close(f"P1 stable fraction k={k}, b={bb:g}", got, rep[str(k)][key], atol=half_cell, rtol=0)


def owen_checks(ck, S):
    O = C.OWEN
    # O2, exactly as sexconflict/experiments/owen_exp.py::worked_example
    o2 = S["owen"]["O2 Parsons example"]
    a1, h1, b1, a2, h2, b2 = O.example
    ck.close("O2 latent root lambda", op.autosomal_invasion_eigenvalue(h1, b1, h2, b2), o2["lambda_"])
    traj = op.iterate(op.autosomal_step, 0.01, 0.01, O.example, n_gen=1500, record=True)
    ck.close("O2 deterministic p1 after 1500 gens", traj[-1, 0], o2["det_final"])
    eq = op.autosomal_equilibria(*O.example)
    ck.equal("O2 number of interior equilibria", len(eq), len(o2["equilibria"]))
    for e, (p1, p2, stable) in zip(eq, o2["equilibria"]):
        ck.close("O2 equilibrium p1", e["p1"], p1)
        ck.equal("O2 equilibrium stable", e["stable"], stable)
    # O4, exactly as owen_exp.py::owen_bistability
    o4 = S["owen"]["O4 Owen bistability"]["equilibria"]
    eq = op.autosomal_equilibria(*O.bistable)
    ck.equal("O4 number of interior equilibria", len(eq), len(o4))
    for i, (e, s) in enumerate(zip(eq, o4)):
        ck.close(f"O4 equilibrium {i + 1} p1", e["p1"], s["p1"])
        ck.close(f"O4 equilibrium {i + 1} p2", e["p2"], s["p2"])
        ck.equal(f"O4 equilibrium {i + 1} stable", e["stable"], s["stable"])
    cubic = sorted(round(float(c[0]), 6) for c in op.blueprint_cubic_equilibria(*O.bistable))
    ck.equal("O4 blueprint cubic == numeric p1 (6 dp)", cubic, sorted(round(float(e["p1"]), 6) for e in eq))


def report_check(ck):
    spec = importlib.util.spec_from_file_location("make_report", os.path.join(ROOT, "scripts", "make_report.py"))
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(RES, "summary.json"), tmp)
        mr.RES = tmp
        with contextlib.redirect_stdout(_io.StringIO()):
            mr.main()
        same = filecmp.cmp(os.path.join(tmp, "REPORT.md"), os.path.join(RES, "REPORT.md"), shallow=False)
    ck.equal("RPT REPORT.md regenerates identically", same, True)


def ibm_p0(S, N=500):
    """Informational: P0 IBM at one N, same settings and seeds as poster_exp.py."""
    from sexconflict.ibm import runners as R
    P = C.POSTER
    b = P.b_strong
    lo, hi = pm.polymorphism_range(b)
    errs = []
    for a in np.linspace(lo + 0.004, hi - 0.004, 9):
        w = C.validate_viability(C.poster_viability(a, b, 0, 1))
        out = R.run_haploid([0, N // 2, 0, N - N // 2], w, 0.5, 1500, P.one_locus_reps,
                            C.seed_for("poster/" + f"ol{N}{a:.4f}"))
        errs.append(out["pA1"][:, 500:].mean(1).mean() - float(pm.pstar(a, b)))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    stored = S["poster"]["P0 one-locus polymorphism"]["rmse_by_N"].get(str(N))
    same = stored is not None and abs(rmse - stored) < 1e-12
    print(f"\n[--ibm] P0 IBM, N={N}: RMSE of time-averaged A1 frequency vs p* = {rmse:.12g} "
          f"(stored run: {stored:.12g}; {'identical' if same else 'differs'}; "
          f"backend here: {R.backend().__name__.rsplit('.', 1)[-1]})")
    print("        Informational only: see the module docstring.")


def p4_checks(ck, Ns=(500,)):
    """Replay the stored P4 IBM (poster_exp.py::thresholds) for the given N with the
    current kernels and the stored seeds, and compare the number of resolved
    populations per (a, dose) with results/data/poster_threshold_ibm.parquet.
    The modifier draws share one random generator across all (a, N) blocks, so
    every burn-in and draw is replayed in the original order; only the target
    sizes are simulated. P4 is the only caller of run_haploid(stop_on_m_absorb=True)."""
    import pandas as pd
    os.environ["SC_BACKEND"] = "cpu"          # exact comparison: numba CPU kernels
    from sexconflict.ibm import runners as R
    P = C.POSTER
    b, r = P.b_strong, P.r_bif
    stored = pd.read_parquet(os.path.join(RES, "data", "poster_threshold_ibm.parquet"))
    copies = np.rint(stored.m0 * stored.N).astype(int)
    ck.equal(f"P4 all {len(stored)} stored cells: 0 < M1 founder copies < N",
             bool(np.all((copies > 0) & (copies < stored.N))), True)
    rng = np.random.default_rng(C.seed_for("poster/thr-rng"))
    got, want = [], []
    for a in P.thr_a:
        w = C.validate_viability(C.poster_viability(a, b, *P.thr_k))
        for n_pop in P.thr_N:
            nA1 = R.ibm_iasc_burn_in(n_pop, w, P.thr_reps, C.seed_for("poster/" + f"tburn{a}{n_pop}"))
            for m0 in P.thr_m0:
                init = R.introduce_modifier(nA1, n_pop, int(round(m0 * n_pop)), rng)
                if n_pop not in Ns:
                    continue
                out = R.run_haploid(init, w, r, P.thr_gens, P.thr_reps,
                                    C.seed_for("poster/" + f"thr{a}{n_pop}{m0}"), stop_on_m_absorb=True)
                got.append(int(np.sum(out["pM1"][:, -1] > 0.5)))
                row = stored[(stored.a == a) & (stored.N == n_pop) & (stored.m0 == m0)]
                want.append(int(row.k.iloc[0]))
    same = sum(g == s for g, s in zip(got, want))
    label = ", ".join(map(str, Ns))
    ck.equal(f"P4 IBM N={label}: cells with identical resolved count", f"{same}/{len(want)}",
             f"{len(want)}/{len(want)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ibm", action="store_true", help="also re-run the P0 IBM at N=500 (informational)")
    ap.add_argument("--p4", action="store_true", help="also replay the stored P4 IBM at N=500 (strict, CPU)")
    ap.add_argument("--p4-all", action="store_true", help="replay every stored P4 block, N=500/2000/6000 (slow)")
    args = ap.parse_args()
    with open(os.path.join(RES, "summary.json"), encoding="utf-8") as fh:
        S = json.load(fh)
    ck = Checks()
    poster_checks(ck, S)
    owen_checks(ck, S)
    report_check(ck)
    if args.p4_all:
        p4_checks(ck, C.POSTER.thr_N)
    elif args.p4:
        p4_checks(ck)
    ok = ck.report()
    if args.ibm:
        ibm_p0(S)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

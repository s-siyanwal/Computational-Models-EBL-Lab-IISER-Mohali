"""Regression tests of the coupled model (docs/09_coupling.md).

1. genetic layer off   -> the assay layer reproduces its current outputs
2. assay layer off     -> the genetic layer reproduces the Manas core and the
                          current Manas comparison table
3. an SA locus with no effect on kappa, beta or the fecundity cue leaves
   courts-first, latency and courts-most unchanged
4. male recombination is 0 in every Drosophila run
plus: telegony ignores the stepfather's genotype (SA included).
"""
import ast
import hashlib
import io
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extensions import ModuleOff  # noqa: E402
from extensions import choice as C  # noqa: E402
from extensions import coupler as K  # noqa: E402
from extensions import telegony as T  # noqa: E402
from extensions.config import load_regime, values  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402
from iasc.ibm import regime as Rg  # noqa: E402

KEYS = ("cf_sham", "courted", "lat_sham", "lat_inf", "cm_sham")


@pytest.fixture(scope="module")
def regs():
    ius, khan = load_regime("ius"), load_regime("khan")
    P = values(ius)
    Pk = dict(P)
    Pk.update(values(khan))
    return ius, khan, P, Pk


def _hash(d):
    m = hashlib.sha256()
    for k in sorted(d):
        a = np.asarray(d[k])
        m.update(k.encode())
        m.update(a.astype(str).tobytes() if a.dtype.kind in "UO" else np.ascontiguousarray(a, dtype=float).tobytes())
    return m.hexdigest()


def _same(r1, r2):
    for k in KEYS:
        a, b = np.asarray(r1[k], float), np.asarray(r2[k], float)
        if not np.array_equal(a, b, equal_nan=True):
            return False
    return True


@pytest.fixture(scope="module")
def small_pop():
    """A small evolved I/U pair with SA sites and nonzero assay traits."""
    rng = np.random.default_rng(3)
    arch = A.build(rng, 1024, 256, mix={3: .6, 1: .15, 2: .15, 4: .05, 5: .05}, sa_benefit_ratio=0.5)
    tarch = Rg.add_traits(rng, arch, 64, 16, {t: 0.08 for t in Rg.TRAITS})
    Ga = np.zeros((2 * 200, arch.WA + arch.WX), np.uint64)
    E.evolve_one(200, 200, arch.WA, arch.WX, arch.posA, arch.posX, 2.0, 1.3, 1.0, 0.25, arch.eff_f, arch.eff_m, 1.0,
                 False, 5, Ga)
    anc = Rg.ancestor_reference(Ga, arch, tarch)
    env = Rg.Environment(a0=-np.log(0.7) / 1.5 ** 2.08, cost_decay=1 / 24)
    pops = {}
    for reg in ("I", "U"):
        G0 = Rg.sample(Ga, 200, 60, 60, rng)
        pops[reg] = Rg.run_regime(G0, arch, tarch, Rg.Rates(1.0, 0.66, 0.5, 0.12), reg, 8, env, anc, 11, rng)
    return arch, tarch, env, anc, pops


def _record(arch, tarch, env, anc, pops):
    return pd.concat([Rg.genotype_record(pops[r], arch, tarch, env, anc, {"population": r}) for r in ("I", "U")])


# ------------------------------------------------------------------ 1
def test_genetic_layer_off_reproduces_current_assay_outputs(regs):
    ius, khan, P, Pk = regs
    F = dict(ius["modules"])
    # (a) golden outputs captured before the coupling existed
    assert _hash(C.simulate_chinmay_study(np.random.default_rng(11), P, F)) == \
        "1c3f62f2bebdd404d77af14afdd69c229b028aa2539134f0441ad3d22cc06790"
    k = C.simulate_khan(np.random.default_rng(12), Pk, F)
    assert _hash({kk: np.atleast_1d(v) for kk, v in k.items()}) == \
        "40d61ab3df11fdcdff506be352aeb3cda076c40911c6c9e78a88f6dbdf647355"
    # (b) the coupling flag defaults to off
    rec = pd.DataFrame({"sex": ["F", "M"], "population": ["I", "I"]})
    with pytest.raises(ModuleOff):
        K.simulate_chinmay_study(np.random.default_rng(1), P, F, rec, np.random.default_rng(2))
    # (c) a saved Manas population carries no assay traits: coupled == uncoupled, bit for bit
    sv = K.load_saved_population(ROOT / "results" / "pops" / "lh_shared_default.npz")
    assert sv.kappa_13h.isna().all() and sv.beta_rel.isna().all()
    rec = pd.concat([sv.assign(population="I"), sv.assign(population="U")])
    coupled = K.simulate_chinmay_study(np.random.default_rng(11), P, dict(F, coupling=True), rec, np.random.default_rng(99))
    assert _same(coupled, C.simulate_chinmay_study(np.random.default_rng(11), P, F))


# ------------------------------------------------------------------ 2
def test_assay_layer_off_reproduces_manas_core_and_comparison_table():
    # the genetic layer's U regime with no assay traits is the Manas engine
    rng = np.random.default_rng(1)
    arch = A.build(rng, 1024, 256, mix={3: 0.6, 1: 0.15, 2: 0.15, 4: 0.05, 5: 0.05}, sa_benefit_ratio=0.5)
    out = np.zeros((2 * 60, arch.WA + arch.WX), np.uint64)
    E.evolve_one(60, 40, arch.WA, arch.WX, arch.posA, arch.posX, 1.0, 0.66, 0.4, 0.1, arch.eff_f, arch.eff_m, 1.0, False, 7, out)
    G = Rg.run_regime(np.zeros_like(out), arch, Rg.zero_traits(arch.eff_f.size), Rg.Rates(1.0, 0.66, 0.4, 0.1), "U", 40,
                      Rg.Environment(), Rg.Ancestor(), 7, np.random.default_rng(0))
    assert np.array_equal(G, out)
    # the current Manas comparison table is rebuilt unchanged
    from scripts import make_comparison as M
    comp, _ = M.build_table()
    buf = io.StringIO()
    comp.to_csv(buf, index=False)
    ref = (ROOT / "results" / "comparison_table.csv").read_text(encoding="utf-8")
    assert buf.getvalue().replace("\r\n", "\n") == ref.replace("\r\n", "\n")


# ------------------------------------------------------------------ 3
def test_sa_locus_without_kappa_beta_effect_leaves_choice_statistics_unchanged(regs, small_pop):
    ius, khan, P, Pk = regs
    arch, tarch, env, anc, pops = small_pop
    F = dict(ius["modules"], coupling=True)
    rec1 = _record(arch, tarch, env, anc, pops)
    # same genomes, SA sites' fitness effects changed (x3, sign flipped): adult fitness changes,
    # kappa, beta and the cue do not
    sa = np.isin(arch.kind, (4, 5))
    arch2 = A.Architecture(arch.posA, arch.posX, np.where(sa, -3 * arch.eff_f, arch.eff_f),
                           np.where(sa, -3 * arch.eff_m, arch.eff_m), arch.kind, arch.WA, arch.WX)
    rec2 = _record(arch2, tarch, env, anc, pops)
    assert not np.allclose(rec1.w_f, rec2.w_f) and not np.allclose(rec1.sa_load_m, rec2.sa_load_m)
    assert np.array_equal(rec1.kappa_13h.fillna(-1), rec2.kappa_13h.fillna(-1))
    run = lambda rec, PP=P: K.simulate_chinmay_study(np.random.default_rng(5), PP, F, rec, np.random.default_rng(6))  # noqa: E731
    r1, r2 = run(rec1), run(rec2)
    assert _same(r1, r2)
    a1, a2 = C.analyse_chinmay(r1), C.analyse_chinmay(r2)
    for key in ("cf_sham_share", "cm_mean_all", ("cl", "INF")):
        assert a1[key] == a2[key]
    # the test is not vacuous: changing kappa does change the statistics ...
    rec3 = rec1.assign(kappa_13h=np.where(rec1.sex == "F", 0.9, np.nan))
    assert not _same(r1, run(rec3))
    # ... and the documented third channel (baseline fecundity cue, off by default) lets SA act
    Pc = dict(P, cue_includes_baseline_fecundity=True)
    assert not _same(run(rec1, Pc), run(rec2, Pc))


# ------------------------------------------------------------------ 4
def test_male_recombination_is_zero():
    import tomllib
    core = tomllib.loads((ROOT / "config" / "default.toml").read_text(encoding="utf-8"))
    assert core["drosophila"]["male_recombination"] is False
    rng = np.random.default_rng(4)
    arch = A.build(rng, 1024, 256, mix={1: 0.5, 2: 0.5})
    N = 40
    G0 = rng.integers(0, 2 ** 63, size=(2 * N, arch.WA + arch.WX), dtype=np.uint64)
    G0[2 * np.arange(N // 2, N) + 1, arch.WA:] = 0               # males carry a Y
    with pytest.raises(ValueError):
        Rg.run_regime(G0, arch, Rg.zero_traits(arch.eff_f.size), Rg.Rates(5.0, 2.0, 0.0, 0.0, male_recombination=True),
                      "U", 1, Rg.Environment(), Rg.Ancestor(), 1, rng)
    G1 = Rg.run_regime(G0, arch, Rg.zero_traits(arch.eff_f.size), Rg.Rates(5.0, 2.0, 0.0, 0.0), "U", 1,
                       Rg.Environment(), Rg.Ancestor(), 1, rng)
    WA = arch.WA
    male_haps = {G0[2 * i + s, :WA].tobytes() for i in range(N // 2, N) for s in (0, 1)}
    female_haps = {G0[2 * i + s, :WA].tobytes() for i in range(N // 2) for s in (0, 1)}
    pat = [G1[2 * j + 1, :WA].tobytes() for j in range(N)]
    mat = [G1[2 * j, :WA].tobytes() for j in range(N)]
    assert all(h in male_haps for h in pat)                      # paternal autosomes pass intact
    assert sum(h not in female_haps for h in mat) > N // 2       # females do recombine (R = 5 M)
    # every Drosophila-labelled run in the results has male recombination off;
    # True appears only in replications of the preprint's own (SLiM default) setting
    paper_labels = ("maplen/paper", "mu_sweep/paper", "lambda1", "lambda2", "N625", "N1250")
    for f in (ROOT / "results").glob("*.csv"):
        d = pd.read_csv(f)
        if "male_recombination" not in d.columns:
            continue
        lab = d["label"].astype(str)
        assert not d[lab.str.contains("drosophila")].male_recombination.astype(bool).any(), f.name
        assert lab[d.male_recombination.astype(bool)].isin(paper_labels).all(), f.name


# ------------------------------------------------------------------ telegony
def test_telegony_ignores_stepfather_genotype_including_sa(regs, small_pop):
    ius, khan, P, Pk = regs
    arch, tarch, env, anc, pops = small_pop
    rec = _record(arch, tarch, env, anc, pops)
    sa = rec[rec.sex == "M"].groupby("population").sa_load_m.mean()
    cond = {"I": float(sa["I"]), "U": float(sa["U"]), "S": 0.0}
    F = dict(ius["modules"])
    assert P["compartmentalization"] is True
    d0 = T.simulate_chinmay_telegony(np.random.default_rng(8), P, F)
    d1 = T.simulate_chinmay_telegony(np.random.default_rng(8), P, F, stepfather_condition=cond)
    assert all(np.array_equal(d0[k], d1[k]) for k in d0)
    # the coupler gives telegony nothing: it does not import it
    tree = ast.parse((ROOT / "extensions" / "coupler.py").read_text(encoding="utf-8"))
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any("telegony" in m for m in names)


# ------------------------------------------------------------------ record formulas
def test_kappa_and_hazard_fall_with_clearance_and_match_the_scenario_anchor():
    from scripts.run_coupled import environment, load_cfg
    cc = load_cfg()
    for scen in cc["scenario"]:
        env = environment(cc, scen)
        assert abs(Rg.kappa(1.0, 1.0, env, env.assay_dose, 96.0) - scen["kappa_anc_96h"]) < 1e-12
        cl = np.array([0.5, 1.0, 2.0])
        for t in (13.0, 96.0, 105.0):
            k = Rg.kappa(1.0, cl, env, env.assay_dose, t)
            assert np.all(np.diff(k) <= 1e-15)                      # faster clearance never raises the cost
        assert np.all(np.diff(Rg.hazard96(cl, np.zeros(3, bool), env, 1.5)) < 0)
    # the dose exponent is derived from the two mortality inputs
    e = cc["environment"]
    assert abs(e["nu"] - np.log(np.log(5) / np.log(2)) / np.log(1.5)) < 0.005


# ------------------------------------------------------------------ output labelling
@pytest.mark.parametrize("name", ["coupled_comparison.csv", "coupled_comparison_sens_sd_realised.csv"])
def test_coupled_csv_never_relabels_a_by_construction_null(name):
    path = ROOT / "outputs" / name
    if not path.exists():
        pytest.skip("run scripts/run_coupled.py first")
    d = pd.read_csv(path)
    assert d.evidence_type.notna().all() and (d.evidence_type.str.strip() != "").all()
    hand = d[d.scenario == "kappa0"]
    a1 = hand[hand.id.str.startswith("A1-")]
    assert len(a1) and a1.evidence_type.str.startswith("by construction").all()
    assert not hand.evidence_type.str.contains("out-of-sample|prediction", case=False).any()
    assert hand[hand.id == "A1-explained"].verdict.str.startswith("NO").all()
    # the rejected rows stay rejected / in tension
    for i, word in (("H8b-coupled", "REJECTED"), ("H10-coupled", "REJECTED"), ("H4-coupled", "TENSION"), ("H14-coupled", "TENSION")):
        assert d[d.id == i].verdict.str.contains(word).all(), i

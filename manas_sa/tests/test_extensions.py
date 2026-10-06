"""Acceptance tests for the optional extensions (spec:
mate_choice_chinmay/acceptance_tests_extensions.csv): A1-A5, B1-B3, C1-C4,
plus checks of the hand-written statistics and the config loader.

Stochastic properties are asserted over replicate simulated studies with fixed
seeds (distributional criteria), never on one hand-picked seed.
"""
import ast
import hashlib
import os
import pathlib

import numpy as np
import pytest
from scipy import stats as st

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
ROOT = pathlib.Path(__file__).resolve().parents[1]

from extensions import DEFAULT_FLAGS, ModuleOff  # noqa: E402
from extensions import choice as C  # noqa: E402
from extensions import harm as Hm  # noqa: E402
from extensions import stats as S  # noqa: E402
from extensions import telegony as T  # noqa: E402
from extensions.config import load_regime, parse_yaml_subset, values  # noqa: E402

R = 40          # replicate studies per distributional test


@pytest.fixture(scope="module")
def regs():
    ius, khan = load_regime("ius"), load_regime("khan")
    P = values(ius)
    Pk = dict(P)
    Pk.update(values(khan))
    return ius, khan, P, Pk


def reps(P, flags, n, seed, **kw):
    return [C.analyse_chinmay(C.simulate_chinmay_study(np.random.default_rng(seed + r), P, flags, **kw)) for r in range(n)]


# ------------------------------------------------------------------ plumbing
def test_config_every_parameter_has_provenance(regs):
    ius, khan, P, Pk = regs
    for cfg in (ius, khan):
        for k, v in cfg["parameters"].items():
            assert v["provenance"].strip(), k
    assert P["kappa"] == 0.0 and Pk["kappa"] > 0          # kappa_IUS = 0, kappa_Khan > 0
    assert P["compartmentalization"] is True


def test_yaml_subset_parser():
    d = parse_yaml_subset('a:\n  b:\n    value: 1.5  # c\n    provenance: "x: y # not a comment"\n  c: [1, 2]\nflag: true\n')
    assert d == {"a": {"b": {"value": 1.5, "provenance": "x: y # not a comment"}, "c": [1, 2]}, "flag": True}


def test_modules_default_off():
    assert not any(DEFAULT_FLAGS.values())
    with pytest.raises(ModuleOff):
        C.simulate_chinmay_study(np.random.default_rng(0), {}, DEFAULT_FLAGS)
    with pytest.raises(ModuleOff):
        Hm.invade({}, DEFAULT_FLAGS, 0.0, 0.5)
    with pytest.raises(ModuleOff):
        T.simulate_chinmay_telegony(np.random.default_rng(0), {}, DEFAULT_FLAGS)


def test_stats_against_scipy_and_known_answers():
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(0, 1, 40), rng.normal(0.6, 1, 40)]
    g = np.r_[["a"] * 40, ["b"] * 40]
    F, p, *_ = S.factorial_anova(y, {"G": g})["G"]
    ref = st.f_oneway(y[:40], y[40:])
    assert abs(F - ref.statistic) < 1e-8 and abs(p - ref.pvalue) < 1e-10
    # thesis Table 3.2 p-values are the smaller one-sided tail
    for k, n, p_thesis in C_TABLE_P:
        assert abs(S.binom_min_tail(k, n) - p_thesis) < 1e-5
    t = np.r_[rng.exponential(1, 300), rng.exponential(0.5, 300)]
    e = t < 3
    t = np.minimum(t, 3)
    grp = np.r_[["a"] * 300, ["b"] * 300]
    w = S.cox_wald(t, e, {"G": grp})["G"][0]
    lr = S.logrank(t, e, grp)[0]
    assert w > 50 and lr > 50 and abs(w - lr) / lr < 0.15


C_TABLE_P = [(18, 39, .374629), (21, 32, .055092), (23, 38, .127938), (12, 26, .422509), (13, 32, .188543),
             (17, 34, .567917), (15, 27, .350554), (15, 31, .5), (14, 29, .5), (15, 29, .5), (14, 31, .36005),
             (6, 17, .166153), (9, 18, .592735), (18, 33, .364166), (18, 28, .092467), (12, 19, .179642)]


# ------------------------------------------------------------------ Module A
def test_kernel_is_half_when_qualities_equal():
    for b in (0.0, 1.0, 5.0, 50.0):
        assert C.p_first(0.7, 0.7, b) == pytest.approx(0.5)


def test_A1_chinmay_null(regs):
    ius, khan, P, Pk = regs
    out = reps(P, ius["modules"], R, 101)
    blk = np.array([[a[("cf_p_block", b)] for b in (1, 2, 3, 4)] for a in out])
    assert (blk < 0.05).mean() < 0.10                               # per-block false-positive rate ~ nominal
    assert np.mean((blk > 0.05).all(1)) >= 0.70                     # "CF p > 0.05 in 4 blocks" is the typical outcome
    assert np.median([a[("cl", "INF")][1] for a in out]) > 0.2      # CL infection p > 0.2
    assert np.mean([0.45 <= a["cm_mean_all"] <= 0.55 for a in out]) >= 0.95   # CM mean 0.45-0.55


def test_A2_spec_compliance_with_spec_override(regs):
    """Spec A2 threshold, met only with the spec override (config/spec_a2.yaml),
    which the audit classifies as fitted to the threshold."""
    ius, khan, P, Pk = regs
    Ps = dict(Pk)
    Ps.update(values(load_regime("spec_a2")))
    b = C.beta(Ps, Ps["male_depleted"])
    assert C.p_first(1.0, 1 - Ps["kappa"], b) > 0.60
    Pi = dict(Ps, decapitated=False)
    r = C.simulate_two_choice(np.random.default_rng(7), 6000, 1.0, 1 - Ps["kappa"], b, Pi)
    assert r["cf_sham"][r["courted"]].mean() > 0.60                 # sham courted first in > 60%


def test_A2_khan_direction_and_cd_with_calibrated_beta(regs):
    """With the independently calibrated sensitivity: preference for sham in
    the right direction, copulation duration unaffected. (The spec's 60%
    threshold is NOT met: documented inconsistency, docs/07.)"""
    ius, khan, P, Pk = regs
    b = C.beta(Pk, Pk["male_depleted"])
    assert 0.5 < C.p_first(1.0, 1 - Pk["kappa"], b) < 0.60
    k = [C.simulate_khan(np.random.default_rng(300 + i), Pk, khan["modules"]) for i in range(R)]
    assert np.mean([np.mean(x["bias_scores"]) for x in k]) < 0.5     # group bias towards sham (sign)
    cd = np.array([x["cd_diff"] for x in k])
    assert abs(cd.mean()) < 3 * cd.std(ddof=1) / np.sqrt(len(cd)) + 1e-3   # copulation duration difference ~ 0


def test_config_beta_matches_independent_calibration_record(regs):
    import json
    ius, khan, P, Pk = regs
    f = ROOT / "outputs" / "byrne_rice_calibration.json"
    if not f.exists():
        pytest.skip("run scripts/calibrate_byrne_rice.py first")
    cal = json.loads(f.read_text())["linear"]["calibration"]
    assert abs(P["beta0"] - cal["depleted"]["beta"]) < 0.05
    assert abs(P["beta0"] * P["depletion_nondepleted"] - cal["nondepleted"]["beta"]) < 0.05


def test_A3_prediction_infection_bias_appears(regs):
    ius, khan, P, Pk = regs
    out = reps(P, ius["modules"], 15, 501, kappa=Pk["kappa"])
    assert np.mean([a["cf_sham_share"] for a in out]) > 0.53         # labelled a PREDICTION in the comparison CSV


def test_A4_cue_off_gives_no_bias(regs):
    ius, khan, P, Pk = regs
    out = reps(P, ius["modules"], R, 701, kappa=Pk["kappa"], cue_on=False)
    assert abs(np.mean([a["cf_sham_share"] for a in out]) - 0.5) < 0.02
    blk = np.array([[a[("cf_p_block", b)] for b in (1, 2, 3, 4)] for a in out])
    assert (blk < 0.05).mean() < 0.10


def test_A5_genetic_core_emits_no_courtship_or_telegony_statistic():
    banned = ("courts_first", "courtship", "courts_most", "stepfather", "telegony", "cf_sham", "cm_sham")
    for f in (ROOT / "iasc").rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                assert not any(m.startswith("extensions") for m in mods), f"{f.name} imports extensions"
        low = src.lower()
        assert not any(b in low for b in banned), f"{f.name} mentions a courtship/telegony statistic"
    # and the core's statistic dictionaries carry none of them
    from iasc.ibm import architecture as A
    from iasc.ibm import engine as E
    from iasc.ibm import stats as St
    rng = np.random.default_rng(1)
    arch = A.build(rng, 1024, 0, mix={1: 0.5, 2: 0.5})
    out = np.zeros((1, 2 * 40, arch.WA), np.uint64)
    E.evolve_many(40, 30, arch.WA, 0, arch.posA, arch.posX, 1.0, 0.0, 0.3, 0.0, arch.eff_f, arch.eff_m, 1.0,
                  False, np.array([3]), out)
    keys = set(St.population_stats(out[0], arch, rng))
    assert not any(b in k.lower() for k in keys for b in ("cf", "cl_", "cm", "court", "latency", "stepfather"))


# ------------------------------------------------------------------ Module B
def test_B1_harm_only_mutant_declines(regs):
    ius, khan, P, Pk = regs
    F = dict(ius["modules"], harm=True)
    for eta in (0.1, 0.5, 1.0):
        r = Hm.invade(P, F, 0.0, eta)
        assert r["s_initial"] < 0 and r["freq"][-1] < r["freq"][0]
    with pytest.raises(ValueError):                                  # Johnstone-Keller signs need an explicit flag
        Hm.invade(dict(P, rho=-0.5), F, 0.0, 0.5)


def test_B2_competitive_harm_mutant_invades_and_females_lose(regs):
    ius, khan, P, Pk = regs
    F = dict(ius["modules"], harm=True)
    r = Hm.invade(P, F, 0.5, 0.5)
    assert r["s_initial"] > 0 and r["freq"][-1] > 0.9
    assert r["female_eggs"][-1] < r["female_eggs"][0]
    traj = Hm.invade_stochastic(np.random.default_rng(2), P, F, 0.5, 0.5)
    assert traj[-1] > traj[0]


def test_B3_flags_off_leave_core_outputs_unchanged():
    """Golden values captured from the core BEFORE the extensions existed."""
    import extensions.choice  # noqa: F401  (importing the extensions must not matter)
    from iasc import estimators as Es
    from iasc import params as IP
    from iasc.ibm import architecture as A
    from iasc.ibm import engine as E
    from iasc.ibm import hemiclone as H
    cfg = IP.load()
    rng = np.random.default_rng(123)
    arch = A.build(rng, 2048, 512, mix=cfg["lh"]["mix_shared"], sa_benefit_ratio=0.5)
    out = np.zeros((1, 2 * 120, arch.WA + arch.WX), np.uint64)
    E.evolve_many(120, 200, arch.WA, arch.WX, arch.posA, arch.posX, 1.0, 0.66, 0.4, 0.1,
                  arch.eff_f, arch.eff_m, 1.0, False, np.array([7]), out)
    assert hashlib.sha256(out.tobytes()).hexdigest() == "af98857d15333fbe3e8a957cb46187379cd7ee97658b119f889f095ef424de31"
    bv = H.hemigenome_bv(out[0], arch, np.random.default_rng(5), 39, 1000)
    beta = {(s, r): 0.5 for s in ("Female", "Male") for r in "MEF"}
    noise = dict(beta)
    noise["day"] = 0.1
    stt = Es.paper_stats(Es.line_means(H.run_assay(bv, cfg["assay"], beta, noise, np.random.default_rng(6))))
    for s, v in (("M", 0.4526151176), ("E", 0.3254761893), ("F", 0.5024290128)):
        assert abs(stt[("r_wgmf", s)] - v) < 1e-9


# ------------------------------------------------------------------ Module C
def test_C1_no_stepfather_effect_sex_effect_present(regs):
    ius, khan, P, Pk = regs
    res = [T.analyse_telegony(T.simulate_chinmay_telegony(np.random.default_rng(900 + r), P, ius["modules"]))
           for r in range(R)]
    pop = np.array([a[("wald", "Population")][2] for a in res])
    sex = np.array([a[("wald", "Gender")][2] for a in res])
    assert (pop < 0.05).mean() <= 0.15 and np.median(pop) > 0.2
    assert (sex < 0.05).mean() >= 0.95


def test_C2_stepfather_condition_cannot_act_when_compartmentalised(regs):
    """The unreplicated 6 h daughter contrast (I, U > S) must not be predicted:
    with the Drosophila flag on, even a large stepfather condition signal leaves
    offspring survival bit-identical."""
    ius, khan, P, Pk = regs
    a = T.simulate_chinmay_telegony(np.random.default_rng(11), P, ius["modules"])
    b = T.simulate_chinmay_telegony(np.random.default_rng(11), P, ius["modules"],
                                    stepfather_condition={"I": 3.0, "U": 3.0, "S": -3.0})
    assert np.array_equal(a["time"], b["time"]) and np.array_equal(a["event"], b["event"])


def test_C3_telostylinus_template(regs):
    ius, khan, P, Pk = regs
    off = [T.simulate_telostylinus(np.random.default_rng(40 + r), dict(P, compartmentalization=False), ius["modules"])
           for r in range(60)]
    on = [T.simulate_telostylinus(np.random.default_rng(40 + r), P, ius["modules"]) for r in range(60)]
    assert np.mean([x["p2_observed"] for x in off]) > 0.8               # second male sires most
    assert abs(np.mean([x["effect_first"] for x in off]) - P["semen_effect_sd"]) < 0.1   # first-male condition shifts trait
    assert abs(np.mean([x["effect_second"] for x in off])) < 0.1
    assert abs(np.mean([x["effect_first"] for x in on])) < 0.1         # Drosophila flag: no shift


def test_C4_module_off_allocates_no_state_and_no_mothers_curse():
    f = T.mate_twice("I", "LHst", DEFAULT_FLAGS)
    assert not hasattr(f, "semen_state")
    g = T.mate_twice("I", "LHst", {"telegony": True}, 1.0)
    assert g.semen_state == {"first_male_condition": 1.0}
    tree = ast.parse((ROOT / "extensions" / "telegony.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            assert not any(("mito" in m) or ("curse" in m) or m.startswith("sexconflict") for m in mods)


def test_B2_invasion_boundary_is_where_selection_changes_sign(regs):
    ius, khan, P, Pk = regs
    F = dict(ius["modules"], harm=True)
    for eta in (0.25, 0.5, 1.0):
        m_star = Hm.invasion_threshold_m(P, F, eta)
        assert m_star > 0
        assert Hm.invade(P, F, m_star + 0.02, eta, generations=1)["s_initial"] > 0
        assert Hm.invade(P, F, m_star - 0.02, eta, generations=1)["s_initial"] < 0
    assert Hm.invasion_threshold_m(dict(P, rho=0.0, lam=0.0), F, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_independent_calibration_never_reads_khan_or_chinmay_targets():
    """Leakage guard: in scripts/calibrate_byrne_rice.py the functions that fit
    beta must not reference Khan's Table 5 or any Chinmay table."""
    src = (ROOT / "scripts" / "calibrate_byrne_rice.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fit_funcs = {"share_large_ahead", "invert", "khan_bias"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in fit_funcs:
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert "KHAN_TABLE5" not in names and not any("TABLE_3" in n for n in names), node.name
    # Khan's observed scores are only read in main(), after the prediction is computed
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Name) and n.id == "KHAN_TABLE5"]
    pred_line = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "pred" for t in n.targets))
    assert lines and min(lines) > pred_line

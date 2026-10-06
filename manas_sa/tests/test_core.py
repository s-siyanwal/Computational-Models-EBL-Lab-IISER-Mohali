"""Core tests: engine against population-genetic theory, analytical identities,
exact reproduction of published code/data, and the zero-leakage rule."""
import ast
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

from iasc import estimators as Es  # noqa: E402
from iasc.analytic import modifier_cc as mc  # noqa: E402
from iasc.analytic import twolocus as tl  # noqa: E402
from iasc.ibm import architecture as A  # noqa: E402
from iasc.ibm import engine as E  # noqa: E402

RAW = ROOT.parent / "experimental_papers_manas" / "12862_2022_1992_MOESM3_ESM.xlsx"


# ------------------------------------------------------------ zero leakage
def test_ibm_does_not_import_analytic_layer():
    for f in (ROOT / "iasc" / "ibm").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "analytic" not in node.module, f"{f.name} imports {node.module}"


# ------------------------------------------------------------ engine
def test_crossover_matches_haldane():
    from numba import njit
    S = 4096
    W = S // 64
    pos = np.sort(np.random.default_rng(0).random(S))
    h0 = np.full(W, ~np.uint64(0))
    h1 = np.zeros(W, np.uint64)

    @njit
    def frac(n, i, j, seed):
        np.random.seed(seed)
        out = np.zeros(W, np.uint64)
        rec = 0
        for _ in range(n):
            E._gamete(h0, h1, out, 0, W, 0, pos, 2.0, True)
            bi = (out[i // 64] >> np.uint64(i % 64)) & np.uint64(1)
            bj = (out[j // 64] >> np.uint64(j % 64)) & np.uint64(1)
            rec += bi != bj
        return rec / n
    for i, j in ((100, 600), (10, 4000)):
        d = (pos[j] - pos[i]) * 2.0
        assert abs(frac(60000, i, j, 1) - (1 - np.exp(-2 * d)) / 2) < 0.01


def expected_ne_ratio(noise, n_per_sex, draws=40000, seed=0):
    """Ne/N for Wright-Fisher parent choice with weights exp(N(0, noise)) among
    n_per_sex parents of each sex: 1/(2Ne) = 1/(4 n_e) with 1/n_e = E[sum p_i^2].
    (Tends to exp(-noise^2) for large n; is higher at finite n.)"""
    w = np.exp(np.random.default_rng(seed).normal(0, noise, (draws, n_per_sex)))
    p = w / w.sum(1, keepdims=True)
    return (1 / n_per_sex) / (p ** 2).sum(1).mean()


@pytest.mark.parametrize("noise", [0.0, 1.0])
def test_neutral_diversity_and_ne(noise):
    expected = expected_ne_ratio(noise, 75)
    rng = np.random.default_rng(2)
    S = 32768
    arch = A.build(rng, n_sites_A=S, mix={1: 1.0})
    z = np.zeros(S)
    N, U = 150, 0.25
    reps = 12
    out = np.zeros((reps, 2 * N, S // 64), np.uint64)
    E.evolve_many(N, 3000, S // 64, 0, arch.posA, arch.posX, 1.0, 0.0, U, 0.0, z, z, noise, True,
                  np.arange(reps, dtype=np.int64) + 11, out)
    pis = []
    for r in range(reps):
        b = E.unpack(out[r], np.arange(2 * N), S // 64).astype(float)
        p = b.mean(0)
        pis.append((2 * p * (1 - p)).sum() * 2 * N / (2 * N - 1))
    # pi is noisy between replicates (coalescent variance): accept within 3 SE
    ratio = np.mean(pis) / (4 * N * U) / expected
    se = np.std(pis, ddof=1) / np.sqrt(reps) / (4 * N * U) / expected
    assert abs(ratio - 1) < max(0.08, 3 * se), (ratio, se)


# ------------------------------------------------------------ analytic
def test_sex_limited_selection_creates_negative_ld_unless_both_sexes_recombine_freely():
    F, M = tl.sex_limited_tables(0.1, 0.1)
    res = {}
    for rf, rm in ((0.1, 0.0), (0.5, 0.0), (0.5, 0.5)):
        x = np.full(4, 0.25)
        y = x.copy()
        for _ in range(40):
            x, y = tl.step_autosomal(x, y, F, M, rf, rm)
        res[(rf, rm)] = 0.5 * (tl.ld(x) + tl.ld(y))
    assert res[(0.1, 0.0)] < res[(0.5, 0.0)] < 0
    assert abs(res[(0.5, 0.5)]) < 1e-12


def test_free_recombination_neutral_ld_halves():
    x = np.array([0.5, 0, 0, 0.5])
    y = x.copy()
    x2, _ = tl.step_autosomal(x, y, np.ones((4, 4)), np.ones((4, 4)), 0.5, 0.5)
    assert abs(tl.ld(x2) - 0.125) < 1e-12


# ------------------------------------------------------------ published code / data
def test_modifier_reproduces_k2_equal_one_claim():
    cases = mc.build_cases(False)
    frac, _ = mc.run_grid(np.array([[0.0, 1.0], [0.5, 1.0]]), cases, False)
    assert frac[:, 2].min() > 0.85             # paper: >85% for k2 = 1, any k1
    assert abs(frac[0, 2] - 0.957) < 0.02      # paper: 95.7% (we get 0.972)


@pytest.mark.skipif(not RAW.exists(), reason="raw data not present")
def test_estimators_recompute_published_values():
    raw = pd.read_excel(RAW, header=1)
    raw.columns = ["Sex", "SexRatio", "Day", "Line", "Fitness"]
    raw = raw.dropna(subset=["Fitness"])
    raw["Fitness"] = raw.Fitness.astype(float)
    st = Es.paper_stats(Es.line_means(raw))
    # exact to the published 4 decimals (requires untransformed male fitness, flag F12)
    for s, rep in (("M", 0.3805), ("E", 0.4027), ("F", 0.2515)):
        assert abs(st[("r_wgmf", s)] - rep) < 6e-5
    for s, rep in (("M", 0.3097), ("E", 0.2986), ("F", 0.3742)):
        assert abs(st[("prop_SA", s)] - rep) < 6e-5
    for k, rep in ((("r_female_across", "M-F"), 0.7688), (("r_male_across", "M-F"), 0.5567),
                   (("r_male_across", "M-E"), 0.6995), (("r_male_across", "F-E"), 0.5415)):
        assert abs(st[k] - rep) < 6e-5

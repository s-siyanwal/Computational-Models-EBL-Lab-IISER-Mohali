"""Unit tests for core functions of Project 2 (manas_sa/iasc): the estimators
of Geeta Arun et al. (2022) and the deterministic two-locus recursion.

Assertions are exact identities or properties that follow from the
definitions; no value is copied from the stored results.
"""
import numpy as np
import pandas as pd
import pytest

from iasc import estimators as Es
from iasc.analytic import twolocus as tl

rng = np.random.default_rng(7)


# ================================================================ estimators
def _line_means(f, m, sr="E"):
    """Long-format line means, as returned by Es.line_means."""
    n = len(f)
    return pd.DataFrame({"Sex": ["Female"] * n + ["Male"] * n, "SexRatio": sr,
                         "Line": list(range(n)) * 2, "w": np.r_[f, m]})


def test_sa_proportion_is_a_function_of_r_for_standardised_line_means():
    # identity stated in the manual (ch. 0b): prop_SA = (1 - r) / 2 exactly
    for _ in range(20):
        f = rng.normal(1.0, 0.1, 39)
        m = 0.4 * f + rng.normal(1.0, 0.2, 39)
        st = Es.paper_stats(_line_means(f, m))
        assert st[("prop_SA", "E")] == pytest.approx((1 - st[("r_wgmf", "E")]) / 2, abs=1e-12)


def test_r_and_sa_proportion_at_the_limits():
    f = rng.normal(1.0, 0.1, 39)
    conc = Es.paper_stats(_line_means(f, 2 * f + 3))        # perfectly concordant lines
    anti = Es.paper_stats(_line_means(f, 5 - f))            # perfectly antagonistic lines
    assert conc[("r_wgmf", "E")] == pytest.approx(1.0)
    assert conc[("prop_SA", "E")] == pytest.approx(0.0, abs=1e-12)
    assert anti[("r_wgmf", "E")] == pytest.approx(-1.0)
    assert anti[("prop_SA", "E")] == pytest.approx(1.0)


def test_line_means_are_invariant_to_a_day_effect():
    rows = [(sex, "E", day, line, rng.uniform(0.2, 1.0))
            for day in (1, 2) for line in range(5) for sex in ("Female", "Male") for _ in range(3)]
    raw = pd.DataFrame(rows, columns=["Sex", "SexRatio", "Day", "Line", "Fitness"])
    scaled = raw.copy()
    scaled.loc[scaled.Day == 2, "Fitness"] *= 7.5        # a day effect multiplies every vial that day
    assert np.allclose(Es.line_means(raw).w, Es.line_means(scaled).w, atol=1e-12)


# ===================================================== two-locus recursion
def _le(pA, pB):
    """Haplotype frequencies [A1B1, A1B2, A2B1, A2B2] at linkage equilibrium."""
    return np.array([pA * pB, pA * (1 - pB), (1 - pA) * pB, (1 - pA) * (1 - pB)])


@pytest.mark.parametrize("r", [0.0, 0.1, 0.5])
def test_recombination_tensor_is_a_probability_distribution(r):
    G = tl._recomb_tensor(r)
    assert np.allclose(G.sum(-1), 1.0)
    # double heterozygote A1B1 / A2B2: parental types (1 - r)/2 each, recombinants r/2 each
    assert np.allclose(G[0, 3], [(1 - r) / 2, r / 2, r / 2, (1 - r) / 2])


@pytest.mark.parametrize("r", [0.0, 0.2, 0.5])
def test_neutral_step_keeps_allele_frequencies_and_decays_ld(r):
    ones = np.ones((4, 4))
    x = rng.dirichlet(np.ones(4))
    xn, yn = tl.step_autosomal(x, x, ones, ones, r, r)
    assert np.allclose(tl.allele1(xn), tl.allele1(x))
    assert tl.ld(xn) == pytest.approx((1 - r) * tl.ld(x), abs=1e-14)
    assert np.allclose(xn, yn)


@pytest.mark.parametrize("r", [0.0, 0.1, 0.3, 0.5])
def test_sex_differential_ld_formula_matches_the_recursion(r):
    # Egg and sperm pools each at linkage equilibrium but with different allele
    # frequencies: one neutral round of gamete formation creates
    # D' = (1 - 2r) dA dB / 4 (sex_differential_ld_one_step), with r = 0 in males.
    ones = np.ones((4, 4))
    for _ in range(10):
        pAx, pBx, pAy, pBy = rng.uniform(0.05, 0.95, 4)
        xn, yn = tl.step_autosomal(_le(pAx, pBx), _le(pAy, pBy), ones, ones, r, 0.0)
        assert tl.ld(xn) == pytest.approx(tl.sex_differential_ld_one_step(pAx - pAy, pBx - pBy, r), abs=1e-14)
        assert tl.ld(yn) == pytest.approx(tl.sex_differential_ld_one_step(pAx - pAy, pBx - pBy, 0.0), abs=1e-14)


def test_mutation_conserves_frequency_and_scales_allele_one():
    x = rng.dirichlet(np.ones(4), 50)
    muA, muB = 1e-3, 3e-4
    xm = tl.mutate(x, (muA, muB))
    pA, pB = tl.allele1(x)
    pAm, pBm = tl.allele1(xm)
    assert np.allclose(xm.sum(-1), 1.0) and np.all(xm >= 0)
    assert np.allclose(pAm, pA * (1 - muA)) and np.allclose(pBm, pB * (1 - muB))
    assert np.array_equal(tl.mutate(x, (0.0, 0.0)), x)


def test_xlinked_sons_carry_the_maternal_x():
    x, y = rng.dirichlet(np.ones(4), 2)
    xn, yn = tl.step_xlinked(x, y, np.ones((4, 4)), np.ones(4), 0.2)
    assert np.allclose(yn, x)                     # X-bearing sperm = the sons' X = the egg pool
    assert xn.sum() == pytest.approx(1.0)

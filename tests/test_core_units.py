"""Unit tests for the core functions of Project 1 (sexconflict/).

Every assertion is an exact identity, a comparison with an independent
reference (scipy, finite differences), or a property that follows from the
model's definition (conservation, boundaries, block structure). No expected
value is copied from the stored results.
"""
import os

import numpy as np
import pytest
from numba import njit
from scipy import stats as st

os.environ.setdefault("SC_PROFILE", "quick")

from sexconflict import config as C  # noqa: E402
from sexconflict.analysis import stats as S  # noqa: E402
from sexconflict.deterministic import mothers_curse as mc  # noqa: E402
from sexconflict.deterministic import owen_parsons as op  # noqa: E402
from sexconflict.deterministic import poster_model as pm  # noqa: E402
from sexconflict.ibm import kernels as K  # noqa: E402
from sexconflict.ibm import runners as R  # noqa: E402

rng = np.random.default_rng(2026)


# ======================================================== comparison statistics
@pytest.mark.parametrize("k", [0, 3, 10, 17, 20])
def test_wilson_interval_matches_scipy(k):
    n, z = 20, st.norm.ppf(0.975)
    p, lo, hi = S.wilson(k, n, z=z)
    ci = st.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert p == pytest.approx(k / n)
    assert lo == pytest.approx(ci.low, abs=1e-12)
    assert hi == pytest.approx(ci.high, abs=1e-12)


def test_agreement_truth_table():
    pred = np.array([True, True, False, False, True])
    lo = np.array([0.6, 0.1, 0.1, 0.6, 0.4])
    hi = np.array([0.9, 0.4, 0.4, 0.9, 0.6])
    assert S.agreement(pred, np.zeros(5), lo, hi).tolist() == \
        ["agree", "disagree", "agree", "disagree", "ambiguous"]


def test_loglog_slope_recovers_an_exact_power_law():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    slope, lo, hi = S.loglog_slope(x, 3.0 * x ** -0.5)
    assert slope == pytest.approx(-0.5, abs=1e-12)
    assert lo <= slope <= hi and hi - lo < 1e-9


def test_rmse_and_ks_helpers():
    assert S.rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))
    assert S.ks_vs_point([1.0, 2.0, 3.0, 4.0], 0.0) == {"D": 1.0, "mae": 2.5}
    assert S.ks_vs_point([-1.0, 1.0, np.nan], 0.0) == {"D": 0.5, "mae": 1.0}
    same = S.ks_two_sample(np.arange(10.0), np.arange(10.0))
    assert same["D"] == 0.0 and same["p"] == pytest.approx(1.0)


def test_fit_logistic_recovers_parameters_from_expected_counts():
    # k = n * p_true maximises the binomial likelihood at the true parameters
    x = np.linspace(0.0, 0.6, 13)
    n = np.full_like(x, 400.0)
    k = n / (1 + np.exp(-(x - 0.3) / 0.05))
    x50, s = S.fit_logistic(x, k, n)
    assert x50 == pytest.approx(0.3, abs=2e-3)
    assert s == pytest.approx(0.05, rel=0.02)


# ============================================================ conditions (config)
def test_seed_for_is_deterministic_and_name_specific():
    names = ["poster/a", "poster/b", "owen/a", "curse/a"]
    seeds = [C.seed_for(n) for n in names]
    assert seeds == [C.seed_for(n) for n in names]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2 ** 31 for s in seeds)


def test_poster_viability_table_is_the_oracle_fitness_table():
    for _ in range(20):
        a, b, k1, k2 = rng.uniform(0, 0.5), rng.uniform(0, 0.5), rng.uniform(0, 1), rng.uniform(0, 1)
        w = C.poster_viability(a, b, k1, k2)
        assert np.array_equal(w[0], pm.female_fitness(b, k1, k2))
        assert np.array_equal(w[1], pm.male_fitness(a))


def test_kidwell_table_follows_the_fitness_scheme():
    s_f, s_m, h_f, h_m = 0.1, 0.2, 0.3, 0.6
    w = C.kidwell_autosomal_viability(s_f, s_m, h_f, h_m)   # columns: number of A1 (0, 1, 2), twice
    assert np.allclose(w[0, :3], [1.0, 1 - h_f * s_f, 1 - s_f])   # A1 (male-beneficial) costs females
    assert np.allclose(w[1, :3], [1 - s_m, 1 - h_m * s_m, 1.0])   # and benefits males
    assert np.array_equal(w[:, :3], w[:, 3:])                       # cytoplasm irrelevant
    w2 = C.kidwell_autosomal_viability(s_f, s_m, h_f, h_m, counted="A2")
    assert np.array_equal(w2[:, :3], w[:, 2::-1])


def test_validate_viability_rejects_bad_tables():
    with pytest.raises(ValueError):
        C.validate_viability([[1.0, np.nan], [1.0, 1.0]])
    with pytest.raises(ValueError):
        C.validate_viability([[0.0, 0.0], [1.0, 1.0]])          # no viable female class
    assert C.validate_viability([[1, 0.5], [2, 1]]).dtype == float


# ======================================================== poster two-locus model
def test_poster_step_stays_on_the_simplex():
    x = rng.dirichlet(np.ones(4), 500)
    b = rng.uniform(0.01, 0.5, 500)
    lo, hi = pm.polymorphism_range(b)
    a = rng.uniform(lo, hi)
    k1, k2, r = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500), rng.uniform(0, 0.5, 500)
    xn = pm.step(x, a, b, k1, k2, r)
    assert np.all(xn >= 0) and np.allclose(xn.sum(-1), 1, atol=1e-12)


def test_poster_neutral_recombination_decays_ld_by_one_minus_r():
    x = rng.dirichlet(np.ones(4), 200)
    r = rng.uniform(0, 0.5, 200)
    xn = pm.step(x, 0.0, 0.0, 0.0, 0.0, r)          # a = b = 0: no selection in either sex
    pA, pM, D = pm.summaries(x)
    pAn, pMn, Dn = pm.summaries(xn)
    assert np.allclose(pAn, pA, atol=1e-14) and np.allclose(pMn, pM, atol=1e-14)
    assert np.allclose(Dn, (1 - r) * D, atol=1e-14)


def test_allele_frequency_parametrisation_round_trips():
    pA, pM = rng.uniform(0.05, 0.95, (2, 300))
    bound = np.minimum.reduce([pA * pM, (1 - pA) * (1 - pM), pA * (1 - pM), (1 - pA) * pM])
    D = rng.uniform(-1, 1, 300) * 0.9 * bound
    x = pm.from_allele_freqs(pA, pM, D)
    assert np.all(x >= 0) and np.allclose(x.sum(-1), 1)
    for got, want in zip(pm.summaries(x), (pA, pM, D)):
        assert np.allclose(got, want, atol=1e-14)


@pytest.mark.parametrize("b", [0.02, 0.2, 0.5])
def test_pstar_spans_zero_to_one_over_the_polymorphism_range(b):
    lo, hi = pm.polymorphism_range(b)
    assert pm.pstar(lo, b) == pytest.approx(0.0, abs=1e-12)
    assert pm.pstar(hi, b) == pytest.approx(1.0, abs=1e-12)
    a = np.linspace(lo, hi, 9)[1:-1]
    p = pm.pstar(a, b)
    assert np.all((p > 0) & (p < 1))
    assert np.allclose(pm.one_locus_step(p, a, b), p, atol=1e-14)


def test_leading_eigenvalue_is_max_of_modifier_block_and_restoring_eigenvalue():
    # At the IaSC point the Jacobian is block-triangular, so its spectral radius
    # is max(rho(B), |mu|), with mu = dp'/dp of the one-locus recursion at p*.
    b = 0.2
    lo, hi = pm.polymorphism_range(b)
    a = rng.uniform(lo + 1e-3, hi - 1e-3, 40)
    k1, k2, r = rng.uniform(0, 1, 40), rng.uniform(0, 1, 40), rng.uniform(0.01, 0.5, 40)
    ps, h = pm.pstar(a, b), 1e-6
    mu = (pm.one_locus_step(ps + h, a, b) - pm.one_locus_step(ps - h, a, b)) / (2 * h)
    lead = pm.leading_eigenvalue(a, b, k1, k2, r)
    assert np.allclose(lead, np.maximum(pm.modifier_block_eigenvalue(a, b, k1, k2, r), np.abs(mu)), atol=1e-8)
    assert np.all(np.abs(mu) < 1)          # the one-locus polymorphism is stable inside the range


def test_classify_labels_final_states():
    a, b = 0.2, 0.2
    assert pm.classify(pm.fixed_point(a, b), a, b) == 0          # IaSC: M1 lost, A at p*
    assert pm.classify(np.array([1.0, 0, 0, 0]), a, b) == 1        # resolved
    assert pm.classify(np.full(4, 0.25), a, b) == 2                # anything else


# ==================================================== Parsons / Owen, X-linked
def _fd_jacobian(step, p, params, h=1e-7):
    J = np.zeros((2, 2))
    for j, (d1, d2) in enumerate(((h, 0.0), (0.0, h))):
        fp = np.array(step(p[0] + d1, p[1] + d2, *params))
        fm = np.array(step(p[0] - d1, p[1] - d2, *params))
        J[:, j] = (fp - fm) / (2 * h)
    return J


def test_sexlinked_latent_root_matches_numerical_jacobian():
    for _ in range(20):
        a1, h1, b1, a2, b2 = rng.uniform(0.5, 1.5, 5)
        J = _fd_jacobian(op.sexlinked_step, (0.0, 0.0), (a1, h1, b1, a2, b2))
        lam = np.abs(np.linalg.eigvals(J)).max()
        assert lam == pytest.approx(op.sexlinked_invasion_eigenvalue(h1, b1, a2, b2), abs=1e-6)


def test_sexlinked_invasion_condition_is_latent_root_above_one():
    h1, b1, a2, b2 = rng.uniform(0.5, 1.5, (4, 2000))
    lam = op.sexlinked_invasion_eigenvalue(h1, b1, a2, b2)
    assert np.array_equal(lam > 1, op.sexlinked_invades(h1, b1, a2, b2))


def test_one_locus_maps_keep_frequencies_in_range_and_fix_the_boundaries():
    par_auto, par_x = rng.uniform(0.5, 1.5, 6), rng.uniform(0.5, 1.5, 5)
    p1, p2 = rng.uniform(0, 1, (2, 500))
    for step, par in ((op.autosomal_step, par_auto), (op.sexlinked_step, par_x)):
        n1, n2 = step(p1, p2, *par)
        assert np.all((n1 >= 0) & (n1 <= 1) & (n2 >= 0) & (n2 <= 1))
        for edge in (0.0, 1.0):
            assert step(edge, edge, *par) == pytest.approx((edge, edge), abs=1e-15)


# ============================================================== mother's curse
@pytest.mark.parametrize("s_m", [0.0, 0.3, 0.6, 0.9])
def test_rare_mito_grows_by_one_plus_s_f_whatever_the_male_cost(s_m):
    s_f, x0 = 0.05, 1e-7
    Z1 = mc.step(mc.initial(x0, 0.0), mc.fitness_tables(s_f=s_f, s_m=s_m))
    assert mc.mito_freq(Z1) / x0 == pytest.approx(float(mc.mito_invasion_rate(s_f)), rel=1e-6)


def test_cytonuclear_state_stays_a_probability_distribution():
    Z = mc.initial(0.3, 0.2)
    assert Z.sum() == pytest.approx(1.0)
    assert mc.mito_freq(Z) == pytest.approx(0.3) and mc.nuclear_freq(Z) == pytest.approx(0.2)
    W = mc.fitness_tables(s_f=0.1, s_m=0.4, restore=0.7, h_R=0.5, cost_R=0.05)
    for _ in range(50):
        Z = mc.step(Z, W)
        assert np.all(Z >= 0) and Z.sum() == pytest.approx(1.0)


def test_restorer_spreads_only_when_males_carry_the_harmful_mito():
    W = mc.fitness_tables(s_f=0.0, s_m=0.5, restore=1.0, h_R=0.5, cost_R=0.0)
    assert mc.nuclear_freq(mc.step(mc.initial(1.0, 0.1), W)) > 0.1
    assert mc.nuclear_freq(mc.step(mc.initial(0.0, 0.1), W)) == pytest.approx(0.1, abs=1e-12)


def test_nuclear_invasion_rate_is_parsons_root_with_unit_homozygote():
    wf, wm = rng.uniform(0, 2, (2, 100))
    assert np.allclose(mc.nuclear_invasion_rate(wf, wm), op.autosomal_invasion_eigenvalue(wf, 1.0, wm, 1.0))


# ============================================================ IBM building blocks
def test_founders_have_exact_marginal_counts():
    c = R.founders_haploid(1000, 0.3, 0.6, np.random.default_rng(1))
    assert c.sum() == 1000 and c[0] + c[1] == 300 and c[0] + c[2] == 600


def test_founders_from_haplotypes_round_to_n():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    c = R.founders_from_haplotypes(1001, x)
    assert c.sum() == 1001 and np.all(np.abs(c - x * 1001) < 1)


def test_introduce_modifier_preserves_allele_counts():
    nA1 = np.array([0, 300, 500, 1000])
    out = R.introduce_modifier(nA1, 1000, 20, np.random.default_rng(2))
    assert np.all(out >= 0) and np.all(out.sum(1) == 1000)
    assert np.array_equal(out[:, 0] + out[:, 1], nA1)
    assert np.all(out[:, 0] + out[:, 2] == 20)


@njit
def _seed_numba(seed):
    np.random.seed(seed)


def test_survival_probability_is_weight_over_the_sex_maximum():
    n = 40_000                                                   # individuals per sex x class
    sex = np.repeat(np.array([0, 0, 1, 1], np.int8), n)         # F class 0, F class 1, M class 0, M class 1
    cls = np.tile(np.repeat(np.array([0, 1]), n), 2).astype(np.int64)
    wtab = np.array([[1.0, 0.25], [4.0, 2.0]])                   # survival: F 1 and 1/4, M 1 and 1/2
    out_f, out_m = np.empty(sex.size, np.int64), np.empty(sex.size, np.int64)
    _seed_numba(7)
    nf, nm = K._survivors(sex, cls, wtab, out_f, out_m)
    f, m = out_f[:nf], out_m[:nm]
    assert np.sum(f < n) == n and np.sum(m < 3 * n) == n          # the top class of each sex always survives
    se = lambda p: 5 * np.sqrt(p * (1 - p) / n)
    assert abs(np.sum(f >= n) / n - 0.25) < se(0.25)
    assert abs(np.sum(m >= 3 * n) / n - 0.5) < se(0.5)          # normalised by the male maximum, not 1


def test_haploid_record_holds_frequencies():
    out = R.run_haploid([10, 20, 30, 40], np.ones((2, 4)), 0.5, 5, 3, 11)
    assert out["x"].shape == (3, 6, 4)
    assert np.allclose(out["x"].sum(-1), 1.0)
    assert np.allclose(out["x"][:, 0], [0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize("init", [[0, 50, 0, 50], [50, 0, 50, 0]])
def test_absorbed_modifier_is_carried_forward_from_generation_zero(init):
    # Regression test: founders with no M1 (or only M1) used to leave every
    # later generation as zeros when stop_on_m_absorb=True.
    out = R.run_haploid(init, np.ones((2, 4)), 0.5, 10, 2, 1, stop_on_m_absorb=True)
    assert np.allclose(out["x"], out["x"][:, :1])
    assert np.allclose(out["pA1"], 0.5)


def test_absorbed_modifier_is_carried_forward_after_loss_or_fixation():
    out = R.run_haploid([1, 49, 0, 50], np.ones((2, 4)), 0.5, 1000, 8, 3, stop_on_m_absorb=True)
    pM = out["pM1"]
    assert np.all((pM[:, -1] == 0) | (pM[:, -1] == 1))
    for rep in range(pM.shape[0]):
        g = np.flatnonzero((pM[rep] == 0) | (pM[rep] == 1))[0]
        assert np.allclose(out["x"][rep, g:], out["x"][rep, g])


def test_xlinked_neutral_frequency_is_unbiased():
    # 450 A-bearing X chromosomes among ~1.5 N X chromosomes: founding frequency ~0.3
    out = R.run_xlinked(1000, 450, np.ones((2, 3)), 30, 200, 5)
    p = (2 * out["p_f"][:, -1] + out["p_m"][:, -1]) / 3
    assert abs(np.nanmean(p) - 0.3) < 0.02

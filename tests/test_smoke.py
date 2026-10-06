"""Smoke tests, verification and compatibility checks (run: pytest -q)."""
import ast
import os
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("SC_PROFILE", "quick")

from sexconflict import config as C  # noqa: E402
from sexconflict.deterministic import mothers_curse as mc  # noqa: E402
from sexconflict.deterministic import owen_parsons as op  # noqa: E402
from sexconflict.deterministic import poster_model as pm  # noqa: E402
from sexconflict.ibm import runners as R  # noqa: E402


# ------------------------------------------------------------ zero leakage
def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
    return mods


def test_ibm_never_imports_the_paper_maths():
    for f in (ROOT / "sexconflict" / "ibm").glob("*.py"):
        for m in _imports(f):
            assert "deterministic" not in m, f"{f.name} imports {m}"
            assert m not in ("sympy", "scipy"), f"{f.name} imports maths library {m}"


def test_config_is_independent_of_both_engines():
    for m in _imports(ROOT / "sexconflict" / "config.py"):
        assert "deterministic" not in m and "ibm" not in m


# ------------------------------------------------------- deterministic
def test_pstar_matches_iteration():
    for a in (0.17, 0.2, 0.24):
        p = 0.5
        for _ in range(30000):
            p = pm.one_locus_step(p, a, 0.2)
        assert abs(p - pm.pstar(a, 0.2)) < 1e-9


def test_fixed_point_residual():
    for a in (0.18, 0.21, 0.24):
        x = pm.fixed_point(a, 0.2)
        assert np.abs(pm.step(x, a, 0.2, 0.3, 0.6, 0.1) - x).max() < 1e-12


def test_symbolic_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.uniform(0.17, 0.245)
        k1, k2, r = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.01, 0.5)
        v0 = pm.fixed_point(a, 0.2)[:3]

        def F(v):
            return pm.step(np.array([*v, 1 - sum(v)]), a, 0.2, k1, k2, r)[:3]

        h = 1e-7
        Jfd = np.array([(F(v0 + h * e) - F(v0 - h * e)) / (2 * h) for e in np.eye(3)]).T
        assert np.abs(Jfd - pm.jacobian_at_fixed_point(a, 0.2, k1, k2, r)).max() < 1e-6


def test_neutral_modifier_is_neutral():
    # k1 = 0, k2 = 1: M1 has no effect -> leading modifier eigenvalue exactly 1
    lam = pm.modifier_block_eigenvalue(np.linspace(0.17, 0.24, 7), 0.2, 0.0, 1.0, 0.2)
    assert np.allclose(lam, 1.0, atol=1e-12)


def test_ideal_modifier_always_invades():
    assert pm.stable_fraction(0.2, 1.0, 1.0, 21, 21) == 0.0


def test_parsons_condition_equals_latent_root():
    rng = np.random.default_rng(1)
    h1, b1, h2, b2 = rng.uniform(0.5, 1.5, (4, 1000))
    lam = op.autosomal_invasion_eigenvalue(h1, b1, h2, b2)
    assert np.array_equal(lam > 1, op.autosomal_invades(h1, b1, h2, b2))


def test_parsons_latent_root_matches_numerical_jacobian():
    par = (1.0, 1.04, 1.0, 0.99, 0.97, 1.0)
    J = op.autosomal_jacobian(0.0 + 1e-9, 0.0 + 1e-9, *par)
    lam = np.abs(np.linalg.eigvals(J)).max()
    assert abs(lam - op.autosomal_invasion_eigenvalue(1.04, 1.0, 0.97, 1.0)) < 1e-5


def test_sexlinked_condition_consistent_with_eq_3_6():
    # small effects: exact h1(a2+b2) > 2 b1 b2 agrees with 2 beta1 + beta2 - alpha2 > 0
    rng = np.random.default_rng(2)
    be1, be2, al2 = rng.uniform(-0.01, 0.01, (3, 2000))
    exact = op.sexlinked_invades(1.0, 1 - be1, 1 - al2, 1 - be2)
    approx = 2 * be1 + be2 - al2 > 0
    margin = np.abs(2 * be1 + be2 - al2) > 1e-3
    assert np.array_equal(exact[margin], approx[margin])


def test_blueprint_cubic_matches_numeric_equilibria():
    par = C.OWEN.bistable
    num = sorted(round(e["p1"], 6) for e in op.autosomal_equilibria(*par))
    cub = sorted(round(c[0], 6) for c in op.blueprint_cubic_equilibria(*par))
    assert num == cub and len(num) == 3


def test_mito_neutral_when_no_female_effect():
    W = mc.fitness_tables(s_f=0.0, s_m=0.7)
    Z = mc.iterate(mc.initial(0.3, 0.1), W, 50)
    assert abs(mc.mito_freq(Z[-1]) - 0.3) < 1e-12


def test_negative_viability_rejected():
    with pytest.raises(ValueError):
        C.validate_viability([[1, -0.1, 1, 1], [1, 1, 1, 1]])


# ------------------------------------------------------------- IBM
def test_ibm_neutral_drift_is_unbiased():
    N, reps = 400, 200
    w = np.ones((2, 4))
    out = R.run_haploid([100, 100, 100, 100], w, 0.5, 30, reps, 11)
    # no directional change and LD decays towards 0
    assert abs(out["pA1"][:, -1].mean() - 0.5) < 4 * out["pA1"][:, -1].std() / np.sqrt(reps)
    assert abs(out["pM1"][:, -1].mean() - 0.5) < 4 * out["pM1"][:, -1].std() / np.sqrt(reps)


def test_ibm_recombination_breaks_ld():
    # start in full LD (only A1M1 and A2M2), neutral: D should halve ~each generation at r=0.5
    out = R.run_haploid([1000, 0, 0, 1000], np.ones((2, 4)), 0.5, 6, 32, 5)
    x = out["x"].mean(0)
    D = x[:, 0] * x[:, 3] - x[:, 1] * x[:, 2]
    assert D[0] > 0.24 and D[-1] < 0.25 * 0.5 ** 4


def test_ibm_diploid_neutral_heterozygosity():
    out = R.run_diploid(500, 500, 0, np.ones((2, 6)), 1, 400, 3)
    assert abs(np.nanmean(out["p_zyg"][:, -1]) - 0.5) < 0.01


def test_ibm_maternal_transmission():
    # mito variant in every individual stays fixed; absent stays absent
    for n in (0, 300):
        out = R.run_diploid(300, 0, n, np.ones((2, 6)), 20, 4, 2)
        assert np.all(out["mito"][:, -1] == n / 300)


def test_ibm_reproducible():
    w = C.poster_viability(0.2, 0.2, 0.5, 0.5)
    a = R.run_haploid([50, 50, 50, 50], w, 0.1, 20, 4, 99)["x"]
    b = R.run_haploid([50, 50, 50, 50], w, 0.1, 20, 4, 99)["x"]
    assert np.array_equal(a, b)


def test_closed_form_modifier_block_matches_sympy():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.17, 0.245, 50)
    k1, k2, r = rng.uniform(0, 1, 50), rng.uniform(0, 1, 50), rng.uniform(0, 0.5, 50)
    J = pm.jacobian_at_fixed_point(a, 0.2, k1, k2, r)
    Bs = J[..., [0, 2]][..., [0, 2], :]
    assert np.abs(Bs - pm.modifier_block_closed_form(a, 0.2, k1, k2, r)).max() < 1e-12


def test_r0_limit_any_k1_invades():
    lam = pm.modifier_block_eigenvalue(0.2, 0.2, 0.05, 0.1, 1e-9)
    Wf = 1 + 0.2 * (1 - pm.pstar(0.2, 0.2))
    assert abs(lam - (1 + 0.2 * 0.05 / (2 * Wf))) < 1e-8


def test_fate_fast_matches_converged_reference():
    rng = np.random.default_rng(4)
    ics = rng.dirichlet(np.ones(4), 40)
    A = np.linspace(0.168, 0.248, 5)[:, None]
    X0 = np.broadcast_to(ics, (5, 40, 4))
    xf, _ = pm.fate_fast(X0, A, 0.2, 0.5, 0.5, 0.1, max_gen=40000)
    xn, _ = pm.iterate(X0, A, 0.2, 0.5, 0.5, 0.1, n_gen=40000, tol=1e-15)
    assert np.array_equal(pm.classify(xf, A, 0.2), pm.classify(xn, A, 0.2))
    assert np.abs(xf - xn).max() < 1e-8

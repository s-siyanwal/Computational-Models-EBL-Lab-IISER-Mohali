"""Checks for the extended SA oracle (literature formulas) and the XY kernel."""
import numpy as np

from sexconflict import config as C
from sexconflict.deterministic import owen_parsons as op
from sexconflict.deterministic import sa_classics as sa
from sexconflict.ibm import runners as R

rng = np.random.default_rng(0)
SF, SM = rng.uniform(0.01, 0.5, (2, 5000))
HF, HM = rng.uniform(0.01, 0.99, (2, 5000))


def test_parsons_root_equals_fry_eq3():
    assert np.array_equal(sa.autosomal_protected(SF, SM, HF, HM), sa.fry_eq3(SF, SM, HF, HM))


def test_additive_is_kidwell_wedge():
    kid = (SM / SF > 1 / (1 + SF)) & (SM / SF < 1 / (1 - SF))
    assert np.array_equal(sa.autosomal_protected(SF, SM, 0.5, 0.5), kid)


def test_xlinked_root_equals_rice_eq2():
    assert np.array_equal(sa.xlinked_protected(SF, SM, HF), sa.rice_eq2(SF, SM, HF))


def test_xy_with_free_recombination_is_autosomal():
    x = (0.3, 0.3, 0.3)
    p1 = p2 = 0.3
    for _ in range(80):
        x = sa.xy_step(*x, 0.1, 0.12, 0.3, 0.6, 0.5)
        p1, p2 = op.autosomal_step(p1, p2, *sa.owen_params(0.1, 0.12, 0.3, 0.6))
    assert abs(x[0] - p1) < 1e-12 and abs(x[1] - p2) < 1e-12 and abs(x[2] - p2) < 1e-12


def test_xy_complete_linkage_segregates_alleles():
    pXe, pXs, pY = sa.xy_iterate((0.5, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5, 0.0))
    assert pY > 0.999 and pXe < 0.01


def test_kimura_limits():
    assert abs(sa.kimura_u(0.1, 500, 0.0) - 0.1) < 1e-12
    assert abs(sa.kimura_u(0.1, 500, 1e-12) - 0.1) < 1e-6
    x = 4 * 500 * 0.01
    assert abs(sa.kimura_u(0.1, 500, 0.01) - (1 - np.exp(-x * 0.1)) / (1 - np.exp(-x))) < 1e-12
    assert sa.kimura_u(0.1, 500, -0.01) < 1e-6


def test_diffusion_solver_neutral_exact():
    N, p = 300, 0.2
    T = sa.mean_absorption_time(lambda q: 0 * q, lambda q: q * (1 - q) / N, p)
    assert abs(T / (-2 * N * (p * np.log(p) + (1 - p) * np.log(1 - p))) - 1) < 1e-3


def test_xy_ibm_complete_linkage():
    w = C.xy_viability(0.1, 0.1, 0.5, 0.5)
    out = R.run_xy(2000, 0.5, 0.5, w, 0.0, 300, 8, 1)
    assert np.nanmean(out["pY"][:, -1]) > 0.97 and np.nanmean(out["pX"][:, -1]) < 0.05


def test_xy_ibm_sex_ratio_and_neutral():
    out = R.run_xy(2000, 0.3, 0.3, np.ones((2, 3)), 0.5, 50, 64, 2)
    # neutral: X and Y frequencies drift around the founding value
    assert abs(np.nanmean(out["pX"][:, -1]) - 0.3) < 0.02 and abs(np.nanmean(out["pY"][:, -1]) - 0.3) < 0.03


def test_gp_denoiser_recovers_known_boundary():
    """Synthetic check of M1: noisy samples of a field whose zero-contour is a
    known wedge; the GP must recover it (and must not inflate it between grid
    points when some cells have zero reported variance)."""
    from sexconflict.analysis import ml
    g = np.linspace(0.02, 0.3, 10)
    F, M = np.meshgrid(g, g)
    Xg = np.column_stack([F.ravel(), M.ravel()])
    truth = lambda x: np.tanh(4 * (x[:, 1] / x[:, 0] - 1 / (1 + x[:, 0]))) * np.tanh(4 * (1 / (1 - x[:, 0]) - x[:, 1] / x[:, 0]))
    r = np.random.default_rng(1)
    y = truth(Xg) + r.normal(0, 0.1, len(Xg))
    var = np.where(r.random(len(Xg)) < 0.3, 0.0, 0.01)       # some cells report zero variance
    fs = np.linspace(0.02, 0.3, 60)
    FF, MM = np.meshgrid(fs, fs)
    fine = np.column_stack([FF.ravel(), MM.ravel()])
    m, _, _ = ml.gp_denoise(Xg, y, var, fine)
    assert ml.iou(m > 0, truth(fine) > 0) > 0.6

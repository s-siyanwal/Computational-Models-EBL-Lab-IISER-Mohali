"""Statistical equivalence of the OpenCL (GPU) and numba (CPU) IBM kernels.
Skipped automatically when no OpenCL GPU is available."""
import numpy as np
import pytest
from scipy import stats

from sexconflict import config as C
from sexconflict.ibm import kernels as K

G = pytest.importorskip("sexconflict.ibm.opencl_backend")
if not G.available():
    pytest.skip("no OpenCL GPU", allow_module_level=True)

REPS = 400


def _hap(fn, init, w, r, T, seed):
    x = fn(init, int(init[0].sum()), T, w, r, len(init), seed) if fn is G.haploid_ensemble else \
        fn(init, int(init[0].sum()), T, w, r, len(init), seed, False)
    N = init[0].sum()
    return (x[..., 0] + x[..., 1]) / N, (x[..., 0] + x[..., 2]) / N, x / N


@pytest.mark.parametrize("case", ["selection", "neutral"])
def test_haploid_gpu_matches_cpu(case):
    N, T = 1000, 150
    w = C.poster_viability(0.22, 0.2, 0.5, 0.5) if case == "selection" else np.ones((2, 4))
    init = np.tile(np.array([25, 475, 25, 475], np.int64), (REPS, 1))
    gA, gM, _ = _hap(G.haploid_ensemble, init, w, 0.1, T, 11)
    cA, cM, _ = _hap(K.haploid_ensemble, init, w, 0.1, T, 11)
    for t in (10, 50, T):
        for g, c in ((gA, cA), (gM, cM)):
            assert stats.ks_2samp(g[:, t], c[:, t]).pvalue > 0.001
            se = np.sqrt(g[:, t].var() / REPS + c[:, t].var() / REPS)
            assert abs(g[:, t].mean() - c[:, t].mean()) < 4 * se + 1e-9


def test_haploid_gpu_recombination():
    init = np.tile(np.array([1000, 0, 0, 1000], np.int64), (64, 1))
    _, _, x = _hap(G.haploid_ensemble, init, np.ones((2, 4)), 0.5, 6, 3)
    m = x.mean(0)
    D = m[:, 0] * m[:, 3] - m[:, 1] * m[:, 2]
    assert D[0] > 0.24 and abs(D[1] / D[0] - 0.5) < 0.05 and D[-1] < 0.25 * 0.5 ** 4


def test_diploid_gpu_matches_cpu():
    w = C.owen_viability(0.98, 1.0, 0.95, 0.98, 1.0, 1.02)
    g = G.diploid_ensemble(80, 0, 1000, 200, w, REPS, 5)
    c = K.diploid_ensemble(80, 0, 1000, 200, w, REPS, 5)
    for t in (20, 100, 200):
        a, b = g[:, t, 0] / 2000, c[:, t, 0] / 2000
        assert stats.ks_2samp(a, b).pvalue > 0.001


def test_diploid_gpu_maternal_and_neutral():
    W = np.ones((2, 6))
    g = G.diploid_ensemble(1000, 300, 1000, 100, W, REPS, 9)
    c = K.diploid_ensemble(1000, 300, 1000, 100, W, REPS, 9)
    assert stats.ks_2samp(g[:, -1, 1], c[:, -1, 1]).pvalue > 0.001     # mito drift
    assert stats.ks_2samp(g[:, -1, 0], c[:, -1, 0]).pvalue > 0.001     # nuclear drift
    full = G.diploid_ensemble(0, 1000, 1000, 20, W, 8, 2)
    assert np.all(full[:, -1, 1] == 1000)                               # maternal: stays fixed

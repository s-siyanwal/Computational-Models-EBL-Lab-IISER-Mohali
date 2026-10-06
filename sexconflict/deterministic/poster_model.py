"""Deterministic two-locus haploid model of intra-locus sexual conflict (IaSC)
with an expression modifier -- reconstruction of Samant, Moitra, Sinha & Prasad
(IISER Mohali poster).

Life cycle (inferred from the poster; it reproduces the poster's
p* = 1/2 + (1/b - 1/a)/2 exactly, see `pstar`):

    haploid offspring (haplotype freqs x, identical in both sexes)
      -> sex-specific haploid viability selection
      -> random union of one female and one male haplotype (diploid zygote)
      -> meiosis with recombination r between locus A and locus M
      -> haploid offspring

Haplotype order everywhere: [A1M1, A1M2, A2M1, A2M2] = [x11, x12, x21, x22].

Fitness scheme (poster, panel B):
    males   : 1+a, 1+a, 1,      1
    females : 1+b*k1, 1, 1+b*k2, 1+b

All functions broadcast over numpy arrays of parameters / states, so whole
parameter grids or ensembles of initial conditions are iterated at once.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp

HAPS = ("A1M1", "A1M2", "A2M1", "A2M2")


def male_fitness(a):
    a = np.asarray(a, float)
    one = np.ones_like(a)
    return np.stack([1 + a, 1 + a, one, one], axis=-1)


def female_fitness(b, k1, k2):
    b, k1, k2 = np.broadcast_arrays(*(np.asarray(v, float) for v in (b, k1, k2)))
    one = np.ones_like(b)
    return np.stack([1 + b * k1, one, 1 + b * k2, 1 + b], axis=-1)


# --------------------------------------------------------------------------
# One-locus IaSC (panel A)
# --------------------------------------------------------------------------
def pstar(a, b):
    """Interior equilibrium frequency of A1 (poster: p* = 1/2 + (1/b - 1/a)/2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.5 + 0.5 * (1.0 / b - 1.0 / a)


def polymorphism_range(b):
    """Open interval of a for which -1 < 1/b - 1/a < 1 (stable polymorphism)."""
    return b / (1 + b), b / (1 - b)


def one_locus_step(p, a, b):
    """Single-locus haploid recursion (M absent). Returns next-gen A1 freq."""
    pm = p * (1 + a) / (1 + a * p)
    pf = p / (1 + b * (1 - p))
    return 0.5 * (pm + pf)


# --------------------------------------------------------------------------
# Two-locus map
# --------------------------------------------------------------------------
def step(x, a, b, k1, k2, r):
    """One generation of the two-locus map. x[..., 4] haplotype frequencies."""
    x = np.asarray(x, float)
    wm = male_fitness(a)
    wf = female_fitness(b, k1, k2)
    m = x * wm
    m = m / m.sum(-1, keepdims=True)
    f = x * wf
    f = f / f.sum(-1, keepdims=True)
    r = np.asarray(r, float)[..., None] if np.ndim(r) else r
    # allele marginals in each sex
    fA1, mA1 = f[..., 0] + f[..., 1], m[..., 0] + m[..., 1]
    fM1, mM1 = f[..., 0] + f[..., 2], m[..., 0] + m[..., 2]
    fA = np.stack([fA1, fA1, 1 - fA1, 1 - fA1], -1)
    mA = np.stack([mA1, mA1, 1 - mA1, 1 - mA1], -1)
    fM = np.stack([fM1, 1 - fM1, fM1, 1 - fM1], -1)
    mM = np.stack([mM1, 1 - mM1, mM1, 1 - mM1], -1)
    xn = 0.5 * (1 - r) * (f + m) + 0.5 * r * (fA * mM + mA * fM)
    return xn / xn.sum(-1, keepdims=True)


def iterate(x0, a, b, k1, k2, r, n_gen=5000, tol=1e-13, record=False):
    """Iterate until max |dx| < tol (per element) or n_gen. Returns (x, gens[, traj])."""
    x = np.asarray(x0, float).copy()
    traj = [x.copy()] if record else None
    done = np.zeros(x.shape[:-1], bool)
    gens = np.full(x.shape[:-1], n_gen)
    for g in range(1, n_gen + 1):
        xn = step(x, a, b, k1, k2, r)
        conv = np.abs(xn - x).max(-1) < tol
        newly = conv & ~done
        gens = np.where(newly, g, gens)
        done |= conv
        x = xn
        if record:
            traj.append(x.copy())
        if not record and done.all():
            break
    if record:
        return x, gens, np.array(traj)
    return x, gens


def summaries(x):
    """(freq A1, freq M1, LD D) from haplotype freqs."""
    x = np.asarray(x)
    pA1 = x[..., 0] + x[..., 1]
    pM1 = x[..., 0] + x[..., 2]
    D = x[..., 0] * x[..., 3] - x[..., 1] * x[..., 2]
    return pA1, pM1, D


def from_allele_freqs(pA1, pM1, D=0.0):
    pA1, pM1, D = np.broadcast_arrays(*(np.asarray(v, float) for v in (pA1, pM1, D)))
    return np.stack([pA1 * pM1 + D, pA1 * (1 - pM1) - D,
                     (1 - pA1) * pM1 - D, (1 - pA1) * (1 - pM1) + D], -1)


def fixed_point(a, b):
    """The IaSC fixed point of the poster: x11 = x21 = 0, x12 = p*."""
    p = pstar(a, b)
    z = np.zeros_like(p)
    return np.stack([z, p, z, 1 - p], -1)


# --------------------------------------------------------------------------
# Linear stability (symbolic Jacobian -> numeric)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _symbolic_jacobian():
    """Exact 3x3 Jacobian of (x11,x12,x21) -> next gen, evaluated at the
    fixed point, derived with sympy (no hand algebra). Returns lambdified fn
    and the analytic 2x2 'modifier block' (x11, x21)."""
    a, b, k1, k2, r = sp.symbols("a b k1 k2 r", positive=True)
    x11, x12, x21 = sp.symbols("x11 x12 x21")
    x = [x11, x12, x21, 1 - x11 - x12 - x21]
    wm = [1 + a, 1 + a, 1, 1]
    wf = [1 + b * k1, 1, 1 + b * k2, 1 + b]
    Wm = sum(xi * wi for xi, wi in zip(x, wm))
    Wf = sum(xi * wi for xi, wi in zip(x, wf))
    m = [xi * wi / Wm for xi, wi in zip(x, wm)]
    f = [xi * wi / Wf for xi, wi in zip(x, wf)]
    fA1, mA1 = f[0] + f[1], m[0] + m[1]
    fM1, mM1 = f[0] + f[2], m[0] + m[2]
    n11 = (1 - r) / 2 * (f[0] + m[0]) + r / 2 * (fA1 * mM1 + mA1 * fM1)
    n12 = (1 - r) / 2 * (f[1] + m[1]) + r / 2 * (fA1 * (1 - mM1) + mA1 * (1 - fM1))
    n21 = (1 - r) / 2 * (f[2] + m[2]) + r / 2 * ((1 - fA1) * mM1 + (1 - mA1) * fM1)
    F = sp.Matrix([n11, n12, n21])
    J = F.jacobian([x11, x12, x21])
    p = sp.Rational(1, 2) + (1 / b - 1 / a) / 2
    J0 = sp.simplify(J.subs({x11: 0, x21: 0, x12: p}))
    fn = sp.lambdify((a, b, k1, k2, r), J0, "numpy")
    block = sp.simplify(J0.extract([0, 2], [0, 2]))
    return fn, J0, block


def jacobian_at_fixed_point(a, b, k1, k2, r):
    """Numeric 3x3 Jacobian at the IaSC fixed point (broadcasts)."""
    fns = _elementwise_jacobian()
    a, b, k1, k2, r = np.broadcast_arrays(*(np.asarray(v, float) for v in (a, b, k1, k2, r)))
    out = np.empty(a.shape + (3, 3))
    for i in range(3):
        for j in range(3):
            out[..., i, j] = np.broadcast_to(np.asarray(fns[i][j](a, b, k1, k2, r), float), a.shape)
    return out


@lru_cache(maxsize=None)
def _elementwise_jacobian():
    _, J0, _ = _symbolic_jacobian()
    syms = sp.symbols("a b k1 k2 r", positive=True)
    return [[sp.lambdify(syms, J0[i, j], "numpy") for j in range(3)] for i in range(3)]


def leading_eigenvalue(a, b, k1, k2, r):
    """Spectral radius of the Jacobian at the IaSC fixed point."""
    J = jacobian_at_fixed_point(a, b, k1, k2, r)
    ev = np.linalg.eigvals(J)
    return np.abs(ev).max(-1)


def modifier_block_eigenvalue(a, b, k1, k2, r):
    """Leading eigenvalue of the 2x2 (x11, x21) block = invasion fitness of M1.
    The Jacobian is block-triangular at the fixed point (x11=x21=0 is
    invariant), so this block alone decides whether M1 can invade."""
    J = jacobian_at_fixed_point(a, b, k1, k2, r)
    B = J[..., [0, 2]][..., [0, 2], :]
    tr = B[..., 0, 0] + B[..., 1, 1]
    det = B[..., 0, 0] * B[..., 1, 1] - B[..., 0, 1] * B[..., 1, 0]
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    return tr / 2 + disc


NEUTRAL_TOL = 1e-9  # |lambda| within this of 1 is treated as neutral (cannot invade)


def is_stable(a, b, k1, k2, r):
    return leading_eigenvalue(a, b, k1, k2, r) < 1.0 + NEUTRAL_TOL


def stable_fraction(b, k1, k2, n_a=101, n_r=101):
    """Fraction of the (a, r) rectangle -- a over the polymorphism range,
    r in (0, 0.5] -- for which the IaSC fixed point is stable.
    This is our reading of the poster's 'fraction of the parameter space
    where the fixed point is stable'."""
    lo, hi = polymorphism_range(b)
    A = np.linspace(lo, hi, n_a + 2)[1:-1]
    R = np.linspace(0, 0.5, n_r + 1)[1:]
    AA, RR = np.meshgrid(A, R)
    k1 = np.asarray(k1, float)
    k2 = np.asarray(k2, float)
    shape = np.broadcast(k1, k2).shape
    out = np.empty(shape)
    for idx in np.ndindex(shape):
        lam = modifier_block_eigenvalue(AA, b, np.broadcast_to(k1, shape)[idx],
                                        np.broadcast_to(k2, shape)[idx], RR)
        out[idx] = np.mean(lam < 1.0 + NEUTRAL_TOL)
    return out


# --------------------------------------------------------------------------
# Global analyses
# --------------------------------------------------------------------------
def classify(x, a, b, tol_m=1e-4, tol_p=2e-3):
    """Label final states: 'iasc' (M1 lost, A at p*), 'resolved' (M1 ~ fixed,
    A1 ~ fixed), 'other'."""
    pA1, pM1, _ = summaries(x)
    ps = pstar(a, b)
    iasc = (pM1 < tol_m) & (np.abs(pA1 - ps) < tol_p)
    resolved = (pM1 > 1 - tol_m) & (pA1 > 1 - tol_p)
    return np.where(iasc, 0, np.where(resolved, 1, 2))


def fate(x0, a, b, k1, k2, r, max_gen=60000, eps=1e-9, tol=1e-13):
    """Iterate (vectorised) until, in every element, the modifier is lost or
    fixed (pM1 < eps or > 1-eps) or the state has converged (max|dx| < tol,
    e.g. a neutral modifier sitting on a line of equilibria), or max_gen.
    Returns final x and the generation at which each element stopped."""
    x = np.asarray(x0, float).copy()
    shape = x.shape[:-1]
    gens = np.full(shape, max_gen)
    done = np.zeros(shape, bool)
    for g in range(1, max_gen + 1):
        xn = step(x, a, b, k1, k2, r)
        conv = np.abs(xn - x).max(-1) < tol
        x = xn
        pm = x[..., 0] + x[..., 2]
        ab = (pm < eps) | (pm > 1 - eps) | conv
        gens = np.where(ab & ~done, g, gens)
        done |= ab
        if done.all():
            break
    return x, gens


def min_modifier_frequency(a, b, k1, k2, r, n_bisect=22, max_gen=60000):
    """Minimum initial M1 frequency (D=0, A at p*) that leads to resolution
    (M1 fixed), vectorised over `a`. 0 where the fixed point is unstable
    (any perturbation resolves); NaN where even M1=0.999 fails."""
    a = np.atleast_1d(np.asarray(a, float))
    ps = pstar(a, b)

    def resolves(m0):
        x, _ = fate_fast(from_allele_freqs(ps, m0, 0.0), a, b, k1, k2, r, max_gen=max_gen, stop_on_absorb=True)
        return summaries(x)[1] > 0.5

    lo = np.full(a.shape, 1e-6)
    hi = np.full(a.shape, 0.999)
    r_lo = resolves(lo)
    r_hi = resolves(hi)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        rm = resolves(mid)
        hi = np.where(rm, mid, hi)
        lo = np.where(rm, lo, mid)
    out = hi.copy()
    out[r_lo] = 0.0
    out[~r_hi] = np.nan
    return out


def modifier_block_closed_form(a, b, k1, k2, r):
    """Hand-derived 2x2 invasion matrix B for (x11, x21) at the IaSC point.

    With p = p*, mean fitnesses Wm = 1 + a p, Wf = 1 + b (1 - p) and
    zero-order A1 frequencies after selection phi_m = p(1+a)/Wm, phi_f = p/Wf:
        al = (1+a)/Wm, ga = 1/Wm, be1 = (1+b k1)/Wf, be2 = (1+b k2)/Wf
        B11 = (1-r)(be1+al)/2 + r(phi_f al + phi_m be1)/2
        B12 = r(phi_f ga + phi_m be2)/2
        B21 = r((1-phi_f) al + (1-phi_m) be1)/2
        B22 = (1-r)(be2+ga)/2 + r((1-phi_f) ga + (1-phi_m) be2)/2
    At r = 0 the eigenvalues are (al+be1)/2 = 1 + b k1/(2 Wf) and
    (ga+be2)/2 = 1 - b(1-k2)/(2 Wf): any k1 > 0 invades on the A1 background.
    """
    a, b, k1, k2, r = np.broadcast_arrays(*(np.asarray(v, float) for v in (a, b, k1, k2, r)))
    p = pstar(a, b)
    Wm, Wf = 1 + a * p, 1 + b * (1 - p)
    phm, phf = p * (1 + a) / Wm, p / Wf
    al, ga, be1, be2 = (1 + a) / Wm, 1 / Wm, (1 + b * k1) / Wf, (1 + b * k2) / Wf
    B = np.empty(a.shape + (2, 2))
    B[..., 0, 0] = 0.5 * (1 - r) * (be1 + al) + 0.5 * r * (phf * al + phm * be1)
    B[..., 0, 1] = 0.5 * r * (phf * ga + phm * be2)
    B[..., 1, 0] = 0.5 * r * ((1 - phf) * al + (1 - phm) * be1)
    B[..., 1, 1] = 0.5 * (1 - r) * (be2 + ga) + 0.5 * r * ((1 - phf) * ga + (1 - phm) * be2)
    return B


# --------------------------------------------------------------------------
# Compiled fate iterator (numba): identical map, per-element early exit.
# The numpy `fate` above stays as the reference implementation (see tests).
# --------------------------------------------------------------------------
from numba import njit, prange  # noqa: E402


@njit(cache=True, fastmath=False)
def _step_scalar(x, a, b, k1, k2, r, out):
    wm0, wm1 = 1.0 + a, 1.0
    wf = (1.0 + b * k1, 1.0, 1.0 + b * k2, 1.0 + b)
    Wm = (x[0] + x[1]) * wm0 + (x[2] + x[3]) * wm1
    Wf = x[0] * wf[0] + x[1] * wf[1] + x[2] * wf[2] + x[3] * wf[3]
    m0, m1, m2, m3 = x[0] * wm0 / Wm, x[1] * wm0 / Wm, x[2] / Wm, x[3] / Wm
    f0, f1, f2, f3 = x[0] * wf[0] / Wf, x[1] * wf[1] / Wf, x[2] * wf[2] / Wf, x[3] * wf[3] / Wf
    fA, mA = f0 + f1, m0 + m1
    fM, mM = f0 + f2, m0 + m2
    out[0] = 0.5 * (1 - r) * (f0 + m0) + 0.5 * r * (fA * mM + mA * fM)
    out[1] = 0.5 * (1 - r) * (f1 + m1) + 0.5 * r * (fA * (1 - mM) + mA * (1 - fM))
    out[2] = 0.5 * (1 - r) * (f2 + m2) + 0.5 * r * ((1 - fA) * mM + (1 - mA) * fM)
    out[3] = 0.5 * (1 - r) * (f3 + m3) + 0.5 * r * ((1 - fA) * (1 - mM) + (1 - mA) * (1 - fM))
    s = out[0] + out[1] + out[2] + out[3]
    for i in range(4):
        out[i] /= s


@njit(parallel=True, cache=True)
def _fate_nb(X, a, b, k1, k2, r, max_gen, eps, tol, stop_on_absorb):
    n = X.shape[0]
    Xo = X.copy()
    gens = np.full(n, max_gen, np.int64)
    for e in prange(n):
        x = Xo[e].copy()
        y = np.empty(4)
        for g in range(1, max_gen + 1):
            _step_scalar(x, a[e], b[e], k1[e], k2[e], r[e], y)
            d = 0.0
            for i in range(4):
                dd = abs(y[i] - x[i])
                if dd > d:
                    d = dd
                x[i] = y[i]
            pm = x[0] + x[2]
            if d < tol or (stop_on_absorb and (pm < eps or pm > 1 - eps)):
                gens[e] = g
                break
        Xo[e] = x
    return Xo, gens


def fate_fast(x0, a, b, k1, k2, r, max_gen=60000, eps=1e-9, tol=1e-13, stop_on_absorb=False):
    """Compiled, parallel iteration with per-element stopping.
    stop_on_absorb=False: run each element to convergence (max|dx| < tol) --
        use when the full final state is classified (basins, bifurcations).
    stop_on_absorb=True: also stop once M1 is lost/fixed -- enough when only
        the modifier's fate matters (threshold bisection)."""
    x0 = np.asarray(x0, float)
    shape = np.broadcast_shapes(x0.shape[:-1], *(np.shape(v) for v in (a, b, k1, k2, r)))
    X = np.broadcast_to(x0, shape + (4,)).reshape(-1, 4)
    flat = [np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), shape).ravel()) for v in (a, b, k1, k2, r)]
    Xo, g = _fate_nb(np.ascontiguousarray(X), *flat, int(max_gen), float(eps), float(tol), bool(stop_on_absorb))
    return Xo.reshape(shape + (4,)), g.reshape(shape)

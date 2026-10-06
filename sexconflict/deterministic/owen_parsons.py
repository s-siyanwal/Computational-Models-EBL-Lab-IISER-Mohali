"""Deterministic one-locus models with sex-differential viability.

* Owen (1953) autosomal model, as analysed by Parsons (1961, Heredity 16:103)
  for the initial progress of a new allele.
* Sex-linked locus (Bennett 1957/58, Mandel 1959) -- Parsons section 3.

Viabilities: females (a1, h1, b1), males (a2, h2, b2) for AA, Aa, aa
(sex-linked males: a2 for A, b2 for a).
State: p1 = freq(A) in female gametes, p2 = freq(A) in male gametes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def autosomal_step(p1, p2, a1, h1, b1, a2, h2, b2):
    q1, q2 = 1 - p1, 1 - p2
    AA, Aa, aa = p1 * p2, p1 * q2 + p2 * q1, q1 * q2
    T1 = a1 * AA + h1 * Aa + b1 * aa
    T2 = a2 * AA + h2 * Aa + b2 * aa
    return (a1 * AA + 0.5 * h1 * Aa) / T1, (a2 * AA + 0.5 * h2 * Aa) / T2


def sexlinked_step(p1, p2, a1, h1, b1, a2, b2):
    q1, q2 = 1 - p1, 1 - p2
    AA, Aa, aa = p1 * p2, p1 * q2 + p2 * q1, q1 * q2
    T1 = a1 * AA + h1 * Aa + b1 * aa
    p1n = (a1 * AA + 0.5 * h1 * Aa) / T1
    p2n = a2 * p1 / (a2 * p1 + b2 * q1)  # sons receive the X from the mother
    return p1n, p2n


def iterate(stepfn, p1, p2, params, n_gen=2000, record=False):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    traj = [(p1, p2)]
    for _ in range(n_gen):
        p1, p2 = stepfn(p1, p2, *params)
        if record:
            traj.append((p1, p2))
    if record:
        return np.array(traj)  # (T+1, 2, ...)
    return p1, p2


# ---------------------------------------------------------------- invasion
def autosomal_invasion_eigenvalue(h1, b1, h2, b2):
    """Dominant latent root at p=0 (Parsons eq. 2.4): (h1/b1 + h2/b2)/2."""
    return 0.5 * (np.asarray(h1) / b1 + np.asarray(h2) / b2)


def autosomal_invades(h1, b1, h2, b2):
    """Parsons (2.5): h1*b2 + h2*b1 > 2*b1*b2."""
    return np.asarray(h1) * b2 + np.asarray(h2) * b1 > 2 * np.asarray(b1) * b2


def sexlinked_invasion_eigenvalue(h1, b1, a2, b2):
    """Dominant root of the linearised sex-linked system at p=0:
    lambda^2 - (h1/2b1) lambda - h1 a2/(2 b1 b2) = 0."""
    c = np.asarray(h1) / (2 * np.asarray(b1))
    d = c * np.asarray(a2) / b2
    return 0.5 * (c + np.sqrt(c * c + 4 * d))


def sexlinked_invades(h1, b1, a2, b2):
    """Exact: h1 (a2 + b2) > 2 b1 b2   (approx. 2 beta1 + beta2 - alpha2 > 0, eq. 3.6)."""
    return np.asarray(h1) * (np.asarray(a2) + b2) > 2 * np.asarray(b1) * b2


# ------------------------------------------------------- equilibria (Owen)
def _u_map(u1, u2, a, h, b):
    return (2 * a * u1 * u2 + h * (u1 + u2)) / (h * (u1 + u2) + 2 * b)


def autosomal_equilibria(a1, h1, b1, a2, h2, b2, n_grid=20001):
    """All interior equilibria, found numerically (no closed form assumed).
    Uses gene ratios u = p/q. The female equation is solved for u2(u1); the
    residual of the male equation is root-bracketed on a fine p1 grid."""
    def u2_of_u1(u1):
        den = u1 * (h1 - 2 * a1) - h1
        return u1 * (h1 - h1 * u1 - 2 * b1) / den

    def resid(p1):
        u1 = p1 / (1 - p1)
        u2 = u2_of_u1(u1)
        return u2 - _u_map(u1, u2, a2, h2, b2)

    ps = np.linspace(1e-9, 1 - 1e-9, n_grid)
    with np.errstate(all="ignore"):
        R = resid(ps)
        U2 = u2_of_u1(ps / (1 - ps))
    eq = []
    for i in range(n_grid - 1):
        if not (np.isfinite(R[i]) and np.isfinite(R[i + 1])):
            continue
        if U2[i] <= 0 or U2[i + 1] <= 0:
            continue
        if np.sign(R[i]) != np.sign(R[i + 1]):
            # avoid poles: require continuity of u2
            if abs(U2[i + 1] - U2[i]) > 10 * (1 + abs(U2[i])):
                continue
            p1 = brentq(resid, ps[i], ps[i + 1], xtol=1e-15)
            u2 = u2_of_u1(p1 / (1 - p1))
            p2 = u2 / (1 + u2)
            eq.append((p1, p2))
    out = []
    for p1, p2 in eq:
        n1, n2 = autosomal_step(p1, p2, a1, h1, b1, a2, h2, b2)
        if abs(n1 - p1) < 1e-9 and abs(n2 - p2) < 1e-9:
            J = autosomal_jacobian(p1, p2, a1, h1, b1, a2, h2, b2)
            lam = np.linalg.eigvals(J)
            out.append(dict(p1=p1, p2=p2, eig=lam, stable=bool(np.abs(lam).max() < 1)))
    return out


def autosomal_jacobian(p1, p2, *params, h=1e-7):
    """Central finite-difference Jacobian of the (p1,p2) map."""
    J = np.zeros((2, 2))
    for j, (d1, d2) in enumerate(((h, 0), (0, h))):
        fp = np.array(autosomal_step(p1 + d1, p2 + d2, *params))
        fm = np.array(autosomal_step(p1 - d1, p2 - d2, *params))
        J[:, j] = (fp - fm) / (2 * h)
    return J


def blueprint_cubic_equilibria(a1, h1, b1, a2, h2, b2):
    """Equilibria from the cubic quoted in the earlier AI blueprint
    ('Simulating Population Genetics Models.txt', attributed to Mandel 1971):
        c_i = 2b_i/h_i - 1, C_i = 2a_i/h_i - 1, t = u2/u1,
        t^3 - 3 alpha t^2 + 3 beta t - 1 = 0,
        3alpha = C1 c2 + c1 + C2,  3beta = C2 c1 + c2 + C1,
        u1 = (t - c1)/(1 - C1 t).
    Returned so the test-suite can check it against the numeric solver."""
    c1, c2 = 2 * b1 / h1 - 1, 2 * b2 / h2 - 1
    C1, C2 = 2 * a1 / h1 - 1, 2 * a2 / h2 - 1
    al = (C1 * c2 + c1 + C2) / 3
    be = (C2 * c1 + c2 + C1) / 3
    roots = np.roots([1, -3 * al, 3 * be, -1])
    out = []
    for t in roots:
        if abs(t.imag) > 1e-10 or t.real <= 0:
            continue
        t = t.real
        u1 = (t - c1) / (1 - C1 * t)
        if u1 <= 0:
            continue
        u2 = t * u1
        out.append((u1 / (1 + u1), u2 / (1 + u2)))
    return out

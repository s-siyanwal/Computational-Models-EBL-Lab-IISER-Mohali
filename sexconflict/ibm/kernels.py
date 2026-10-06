"""Individual-based stochastic simulators (IBM) -- the 'emulation' side.

ZERO-LEAKAGE CONTRACT
---------------------
Nothing in this package may import from `sexconflict.deterministic` or encode
any recursion, equilibrium, eigenvalue or expected-frequency formula.
The only shared inputs are the *conditions* of the paper:
  * a viability table (survival weight of each genotype in each sex),
  * the recombination probability between two loci in one meiosis,
  * the mode of inheritance (haploid / autosomal / X-linked / maternal mito),
  * population size and the initial genotypes of individuals.
Each generation is built only from biological events on individuals:
  sex assignment (fair coin) -> survival (Bernoulli draw, weight / max weight)
  -> each offspring picks a random surviving mother and father
  -> gametes formed by Mendelian segregation (+ crossover) -> offspring.
Frequencies are only *measured* (counted) for the record; they never feed back
into the dynamics. `tests/test_leakage.py` enforces the import rule.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


# =========================================================================
# shared primitives
# =========================================================================
@njit(cache=True)
def _survivors(sex, wclass, wtab, out_f, out_m):
    """Bernoulli viability. wtab[sex, class] are survival weights; the
    survival probability is weight / (largest weight in that sex)."""
    mx0 = 0.0
    mx1 = 0.0
    for c in range(wtab.shape[1]):
        if wtab[0, c] > mx0:
            mx0 = wtab[0, c]
        if wtab[1, c] > mx1:
            mx1 = wtab[1, c]
    nf = 0
    nm = 0
    for i in range(sex.shape[0]):
        s = sex[i]
        w = wtab[s, wclass[i]] / (mx0 if s == 0 else mx1)
        if np.random.random() < w:
            if s == 0:
                out_f[nf] = i
                nf += 1
            else:
                out_m[nm] = i
                nm += 1
    return nf, nm


@njit(cache=True)
def _assign_sex(sex):
    for i in range(sex.shape[0]):
        sex[i] = 0 if np.random.random() < 0.5 else 1


# =========================================================================
# 1. haploid two-locus life cycle (Samant et al. poster)
#    genome = (allele at A: 0=A1, 1=A2 ; allele at M: 0=M1, 1=M2)
#    haplotype class = 2*A + M  -> [A1M1, A1M2, A2M1, A2M2]
# =========================================================================
@njit(cache=True)
def _haploid_one(A, M, n_gen, wtab, r, stop_on_m_absorb, rec):
    N = A.shape[0]
    sex = np.empty(N, np.int8)
    cls = np.empty(N, np.int64)
    fem = np.empty(N, np.int64)
    mal = np.empty(N, np.int64)
    A2 = np.empty_like(A)
    M2 = np.empty_like(M)
    last = n_gen
    for g in range(n_gen + 1):
        # --- measurement only: count offspring haplotypes
        for k in range(4):
            rec[g, k] = 0
        for i in range(N):
            rec[g, 2 * A[i] + M[i]] += 1
        if g == n_gen:
            break
        if stop_on_m_absorb:
            m1 = rec[g, 0] + rec[g, 2]
            if m1 == 0 or m1 == N:
                last = g
                break
        # --- biology
        _assign_sex(sex)
        for i in range(N):
            cls[i] = 2 * A[i] + M[i]
        nf, nm = _survivors(sex, cls, wtab, fem, mal)
        if nf == 0 or nm == 0:
            last = -g  # population failure (never seen for sensible N)
            break
        for j in range(N):
            mom = fem[np.random.randint(nf)]
            dad = mal[np.random.randint(nm)]
            # meiosis in the diploid zygote: pick the parental chromosome
            # contributing locus A, crossover with probability r before M
            if np.random.random() < 0.5:
                p, q = mom, dad
            else:
                p, q = dad, mom
            A2[j] = A[p]
            M2[j] = M[q] if np.random.random() < r else M[p]
        A[:] = A2
        M[:] = M2
    # forward-fill if stopped early (state is absorbed for the M locus)
    if last > 0 and last < n_gen:
        for g in range(last + 1, n_gen + 1):
            for k in range(4):
                rec[g, k] = rec[last, k]
    return last


@njit(parallel=True, cache=True)
def haploid_ensemble(init_counts, N, n_gen, wtab, r, n_rep, seed, stop_on_m_absorb):
    """Run n_rep independent populations. init_counts[n_rep, 4] = initial
    number of individuals of each haplotype in each replicate.
    Returns counts[n_rep, n_gen+1, 4]."""
    out = np.zeros((n_rep, n_gen + 1, 4), np.int64)
    for rep in prange(n_rep):
        np.random.seed(seed + rep)
        A = np.empty(N, np.int8)
        M = np.empty(N, np.int8)
        k = 0
        for h in range(4):
            for _ in range(init_counts[rep, h]):
                A[k] = h // 2
                M[k] = h % 2
                k += 1
        _haploid_one(A, M, n_gen, wtab, r, stop_on_m_absorb, out[rep])
    return out


# =========================================================================
# 2. diploid autosomal locus + maternally inherited cytoplasm
#    (Owen/Parsons when the cytoplasm is irrelevant; mother's curse otherwise)
#    class = 3*mito + (number of 'A'/'R' alleles)
# =========================================================================
@njit(cache=True)
def _diploid_one(x1, x2, mt, n_gen, wtab, rec):
    N = x1.shape[0]
    sex = np.empty(N, np.int8)
    cls = np.empty(N, np.int64)
    fem = np.empty(N, np.int64)
    mal = np.empty(N, np.int64)
    y1 = np.empty_like(x1)
    y2 = np.empty_like(x2)
    mt2 = np.empty_like(mt)
    for g in range(n_gen + 1):
        _assign_sex(sex)
        for i in range(N):
            cls[i] = 3 * mt[i] + x1[i] + x2[i]
        nf, nm = _survivors(sex, cls, wtab, fem, mal)
        # measurement only (zygotes and surviving adults)
        _measure_diploid(x1, x2, mt, sex, rec[g], 1, fem, mal, nf, nm)
        if g == n_gen:
            break
        if nf == 0 or nm == 0:
            # population failure (e.g. no males left): freeze the record
            for g2 in range(g + 1, n_gen + 1):
                for k in range(6):
                    rec[g2, k] = rec[g, k]
            break
        for j in range(N):
            mom = fem[np.random.randint(nf)]
            dad = mal[np.random.randint(nm)]
            y1[j] = x1[mom] if np.random.random() < 0.5 else x2[mom]
            y2[j] = x1[dad] if np.random.random() < 0.5 else x2[dad]
            mt2[j] = mt[mom]  # maternal transmission of cytoplasm
        x1[:] = y1
        x2[:] = y2
        mt[:] = mt2


@njit(cache=True)
def _measure_diploid(x1, x2, mt, sex, row, have_adults, fem, mal, nf, nm):
    """row = [A-alleles in zygotes, mito+ zygotes, A-alleles in adult females,
              adult females, A-alleles in adult males, adult males]."""
    N = x1.shape[0]
    s = 0
    c = 0
    for i in range(N):
        s += x1[i] + x2[i]
        c += mt[i]
    row[0] = s
    row[1] = c
    if have_adults > 0:
        sf = 0
        for k in range(nf):
            sf += x1[fem[k]] + x2[fem[k]]
        sm = 0
        for k in range(nm):
            sm += x1[mal[k]] + x2[mal[k]]
        row[2] = sf
        row[3] = nf
        row[4] = sm
        row[5] = nm


@njit(parallel=True, cache=True)
def diploid_ensemble(init_alleles, init_mito, N, n_gen, wtab, n_rep, seed):
    """init_alleles: number of 'A' allele copies among the 2N genes (placed at
    random on chromosomes); init_mito: number of individuals carrying mito 1.
    wtab[2, 6] survival weights by sex and class 3*mito+genotype.
    Returns rec[n_rep, n_gen+1, 6] (see _measure_diploid)."""
    out = np.zeros((n_rep, n_gen + 1, 6), np.int64)
    for rep in prange(n_rep):
        np.random.seed(seed + rep)
        genes = np.zeros(2 * N, np.int8)
        genes[:init_alleles] = 1
        np.random.shuffle(genes)
        x1 = genes[:N].copy()
        x2 = genes[N:].copy()
        mt = np.zeros(N, np.int8)
        mt[:init_mito] = 1
        np.random.shuffle(mt)
        _diploid_one(x1, x2, mt, n_gen, wtab, out[rep])
    return out


# =========================================================================
# 3. X-linked locus (females XX, males XY) -- Parsons section 3
#    females: class = number of A alleles (0..2); males: class = A on X (0/1)
# =========================================================================
@njit(cache=True)
def _xlinked_one(x1, x2, sex, n_gen, wtab, rec):
    N = x1.shape[0]
    cls = np.empty(N, np.int64)
    fem = np.empty(N, np.int64)
    mal = np.empty(N, np.int64)
    y1 = np.empty_like(x1)
    y2 = np.empty_like(x2)
    for g in range(n_gen + 1):
        for i in range(N):
            # males carry a single X (x1); x2 is ignored for them
            cls[i] = (x1[i] + x2[i]) if sex[i] == 0 else x1[i]
        nf, nm = _survivors(sex, cls, wtab, fem, mal)
        sf = 0
        for k in range(nf):
            sf += x1[fem[k]] + x2[fem[k]]
        sm = 0
        for k in range(nm):
            sm += x1[mal[k]]
        rec[g, 0] = sf
        rec[g, 1] = nf
        rec[g, 2] = sm
        rec[g, 3] = nm
        if g == n_gen or nf == 0 or nm == 0:
            break
        newsex = np.empty(N, np.int8)
        for j in range(N):
            mom = fem[np.random.randint(nf)]
            dad = mal[np.random.randint(nm)]
            egg = x1[mom] if np.random.random() < 0.5 else x2[mom]
            if np.random.random() < 0.5:   # daughter: X from mother and father
                newsex[j] = 0
                y1[j] = egg
                y2[j] = x1[dad]
            else:                          # son: X from mother, Y from father
                newsex[j] = 1
                y1[j] = egg
                y2[j] = 0
        x1[:] = y1
        x2[:] = y2
        sex[:] = newsex


@njit(parallel=True, cache=True)
def xlinked_ensemble(init_copies, N, n_gen, wtab, n_rep, seed):
    """init_copies A-bearing X chromosomes are placed at random among all X
    chromosomes of the founding generation (2 per female, 1 per male).
    wtab[2, 3]: females by genotype, males by allele (cols 0,1).
    rec[n_rep, n_gen+1, 4] = [A in adult females, n females, A in adult males, n males]."""
    out = np.zeros((n_rep, n_gen + 1, 4), np.int64)
    for rep in prange(n_rep):
        np.random.seed(seed + rep)
        x1 = np.zeros(N, np.int8)
        x2 = np.zeros(N, np.int8)
        sex = np.empty(N, np.int8)
        _assign_sex(sex)
        slots = np.empty(2 * N, np.int64)
        ns = 0
        for i in range(N):
            slots[ns] = 2 * i
            ns += 1
            if sex[i] == 0:
                slots[ns] = 2 * i + 1
                ns += 1
        sl = slots[:ns].copy()
        np.random.shuffle(sl)
        for k in range(min(init_copies, ns)):
            if sl[k] % 2 == 0:
                x1[sl[k] // 2] = 1
            else:
                x2[sl[k] // 2] = 1
        _xlinked_one(x1, x2, sex, n_gen, wtab, out[rep])
    return out


# =========================================================================
# 4. SA locus on sex chromosomes with genetic sex determination (XY)
#    Each individual carries a maternal X (allele xm) and a paternal sex
#    chromosome (allele xp) that is an X (daughter) or a Y (son).
#    Sex is NOT a coin flip here: it is the sex chromosome the father passed.
#    In fathers, the SA allele travels with the transmitted sex chromosome
#    unless a crossover (prob r) separates it from the sex-determining region.
#    class = number of 'A1' alleles (0..2) for both sexes.
# =========================================================================
@njit(cache=True)
def _xy_one(xm, xp, isY, n_gen, wtab, r, rec):
    N = xm.shape[0]
    cls = np.empty(N, np.int64)
    fem = np.empty(N, np.int64)
    mal = np.empty(N, np.int64)
    ym = np.empty_like(xm)
    yp = np.empty_like(xp)
    yY = np.empty_like(isY)
    for g in range(n_gen + 1):
        # measurement only: A1 on X chromosomes and on Y chromosomes
        aX = 0
        nX = 0
        aY = 0
        nY = 0
        for i in range(N):
            aX += xm[i]
            nX += 1
            if isY[i] == 1:
                aY += xp[i]
                nY += 1
            else:
                aX += xp[i]
                nX += 1
            cls[i] = xm[i] + xp[i]
        rec[g, 0] = aX
        rec[g, 1] = nX
        rec[g, 2] = aY
        rec[g, 3] = nY
        if g == n_gen:
            break
        nf, nm = _survivors(isY, cls, wtab, fem, mal)
        if nf == 0 or nm == 0:
            for g2 in range(g + 1, n_gen + 1):
                for k in range(4):
                    rec[g2, k] = rec[g, k]
            break
        for j in range(N):
            mom = fem[np.random.randint(nf)]
            dad = mal[np.random.randint(nm)]
            ym[j] = xm[mom] if np.random.random() < 0.5 else xp[mom]
            toY = 1 if np.random.random() < 0.5 else 0
            cross = np.random.random() < r
            if toY == 1:   # son: receives the Y (allele xp) unless crossover
                yp[j] = xm[dad] if cross else xp[dad]
            else:          # daughter: receives the X (allele xm) unless crossover
                yp[j] = xp[dad] if cross else xm[dad]
            yY[j] = toY
        xm[:] = ym
        xp[:] = yp
        isY[:] = yY


@njit(parallel=True, cache=True)
def xy_ensemble(pX0, pY0, N, n_gen, wtab, r, n_rep, seed):
    """Founders: sex chromosome of each individual's father drawn 50:50; A1
    placed on each X with prob pX0 and on each Y with prob pY0.
    rec[n_rep, n_gen+1, 4] = [A1 on X, #X, A1 on Y, #Y] among zygotes."""
    out = np.zeros((n_rep, n_gen + 1, 4), np.int64)
    for rep in prange(n_rep):
        np.random.seed(seed + rep)
        xm = np.zeros(N, np.int8)
        xp = np.zeros(N, np.int8)
        isY = np.zeros(N, np.int8)
        for i in range(N):
            isY[i] = 1 if np.random.random() < 0.5 else 0
            xm[i] = 1 if np.random.random() < pX0 else 0
            xp[i] = 1 if np.random.random() < (pY0 if isY[i] == 1 else pX0) else 0
        _xy_one(xm, xp, isY, n_gen, wtab, r, out[rep])
    return out

"""Multilocus individual-based forward simulator (numba), sex-specific selection.

Biology represented
-------------------
* diploid, XX/XY sexes; N/2 females and N/2 males each generation
  (SLiM 'Wright-Fisher' mode with sexes): every offspring has a mother drawn
  with probability proportional to female absolute fitness and a father drawn
  proportional to male absolute fitness.
* one autosome (finite-sites bitset, S_A sites at uniform random positions on
  a map of length R_A Morgans) and optionally an X chromosome (S_X sites, R_X).
* crossovers: Poisson(R) per meiosis at uniform positions. Females recombine;
  males recombine only if male_recombination=True (False = Drosophila).
  X: mothers recombine their two X; fathers transmit their single X to
  daughters and a Y (no sites) to sons.
* mutation: Poisson(U) new derived alleles per gamete at uniformly chosen
  sites (finite sites: a hit on an already-derived site leaves it derived).
* each site carries per-copy effects (e_f, e_m) on female / male relative
  fitness; breeding value = sum over carried derived copies (hemizygous X in
  males counts once). Relative fitness = BV + N(0, noise_sd^2); absolute
  fitness = exp(relative), exactly the model of Geeta Arun (2025, bioRxiv).

Nothing here encodes any recursion or LD formula; LD arises from drift,
selection and recombination among individuals.

Genome slots: genomes[2*i] = maternal haplotype, genomes[2*i+1] = paternal;
individuals 0..N/2-1 are female, N/2..N-1 male. For males the paternal X
words are always zero (Y).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

DEBRUIJN = np.array([0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4,
                     62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5,
                     63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11,
                     46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6], np.int64)
MAGIC = np.uint64(0x03F79D71B4CB0A89)


@njit(cache=True)
def _ctz(w):
    lsb = w & (~w + np.uint64(1))
    return DEBRUIJN[np.int64((lsb * MAGIC) >> np.uint64(58))]


@njit(cache=True)
def _bv(genome, w0, w1, eff):
    """Sum of eff over derived sites set in words [w0, w1) of one haplotype."""
    s = 0.0
    for wi in range(w0, w1):
        w = genome[wi]
        while w != np.uint64(0):
            b = _ctz(w)
            s += eff[wi * 64 + b]
            w &= w - np.uint64(1)
    return s


@njit(cache=True)
def _gamete(h0, h1, out, w0, w1, site0, pos, R, recombine):
    """Write a recombinant of haplotypes h0/h1 (words [w0,w1), sites from
    site0) into out. Crossover count ~ Poisson(R) if recombine."""
    nw = w1 - w0
    k = np.random.poisson(R) if recombine else 0
    start_src = 0 if np.random.random() < 0.5 else 1
    if k == 0:
        src = h0 if start_src == 0 else h1
        for i in range(w0, w1):
            out[i] = src[i]
        return
    # breakpoints as site indices (local), sorted
    nsites = nw * 64
    bps = np.empty(k, np.int64)
    for j in range(k):
        c = np.random.random()
        # first local site with pos >= c (pos sorted, local array)
        lo, hi = 0, nsites
        while lo < hi:
            mid = (lo + hi) // 2
            if pos[site0 + mid] < c:
                lo = mid + 1
            else:
                hi = mid
        bps[j] = lo
    bps.sort()
    cur = start_src
    prev = 0
    for j in range(k + 1):
        end = bps[j] if j < k else nsites
        if end > prev:
            src = h0 if cur == 0 else h1
            a, b = prev, end              # copy local sites [a, b)
            wa, wb = a // 64, (b - 1) // 64
            for wi in range(wa, wb + 1):
                lo_bit = a - wi * 64 if wi == wa else 0
                hi_bit = b - wi * 64 if wi == wb else 64
                if lo_bit == 0 and hi_bit == 64:
                    mask = ~np.uint64(0)
                else:
                    width = hi_bit - lo_bit
                    m = (np.uint64(1) << np.uint64(width)) - np.uint64(1) if width < 64 else ~np.uint64(0)
                    mask = m << np.uint64(lo_bit)
                g = w0 + wi
                out[g] = (out[g] & ~mask) | (src[g] & mask)
        cur = 1 - cur
        prev = end


@njit(cache=True)
def _mutate(out, w0, w1, U):
    n = np.random.poisson(U)
    nsites = (w1 - w0) * 64
    for _ in range(n):
        s = np.random.randint(nsites)
        out[w0 + s // 64] |= np.uint64(1) << np.uint64(s % 64)


@njit(cache=True)
def _fitness(genomes, N, WA, WX, eff_f, eff_m, noise_sd, bvf, bvm, wabs):
    nf = N // 2
    for i in range(N):
        g0 = genomes[2 * i]
        g1 = genomes[2 * i + 1]
        if i < nf:
            v = _bv(g0, 0, WA + WX, eff_f) + _bv(g1, 0, WA + WX, eff_f)
            bvf[i] = v
        else:
            # males: autosome both copies; X maternal copy only (hemizygous)
            v = _bv(g0, 0, WA + WX, eff_m) + _bv(g1, 0, WA, eff_m)
            bvm[i] = v
        wabs[i] = np.exp(v + noise_sd * np.random.standard_normal())


@njit(cache=True)
def _draw(cum, lo, n):
    """Draw index in [lo, lo+n) with prob proportional to cum increments."""
    u = np.random.random() * cum[lo + n - 1]
    a, b = lo, lo + n - 1
    while a < b:
        m = (a + b) // 2
        if cum[m] < u:
            a = m + 1
        else:
            b = m
    return a


@njit(cache=True)
def evolve_one(N, n_gen, WA, WX, posA, posX, RA, RX, UA, UX, eff_f, eff_m, noise_sd,
               male_recomb, seed, genomes_out):
    np.random.seed(seed)
    W = WA + WX
    G = np.zeros((2 * N, W), np.uint64)
    G2 = np.zeros((2 * N, W), np.uint64)
    bvf = np.zeros(N)
    bvm = np.zeros(N)
    wabs = np.zeros(N)
    cumf = np.zeros(N)
    nf = N // 2
    posAll = np.concatenate((posA, posX))
    for gen in range(n_gen):
        _fitness(G, N, WA, WX, eff_f, eff_m, noise_sd, bvf, bvm, wabs)
        acc = 0.0
        for i in range(nf):
            acc += wabs[i]
            cumf[i] = acc
        acc = 0.0
        for i in range(nf, N):
            acc += wabs[i]
            cumf[i] = acc
        for j in range(N):
            mom = _draw(cumf, 0, nf)
            dad = _draw(cumf, nf, N - nf)
            mat = G2[2 * j]
            pat = G2[2 * j + 1]
            # maternal haplotype: recombined autosome + recombined X
            _gamete(G[2 * mom], G[2 * mom + 1], mat, 0, WA, 0, posAll, RA, True)
            if WX > 0:
                _gamete(G[2 * mom], G[2 * mom + 1], mat, WA, W, WA * 64, posAll, RX, True)
            _mutate(mat, 0, WA, UA)
            if WX > 0:
                _mutate(mat, WA, W, UX)
            # paternal haplotype: autosome (no crossover unless male_recomb)
            _gamete(G[2 * dad], G[2 * dad + 1], pat, 0, WA, 0, posAll, RA, male_recomb)
            _mutate(pat, 0, WA, UA)
            if WX > 0:
                if j < nf:      # daughter receives father's X (his maternal slot)
                    for wi in range(WA, W):
                        pat[wi] = G[2 * dad][wi]
                    _mutate(pat, WA, W, UX)
                else:           # son receives Y
                    for wi in range(WA, W):
                        pat[wi] = np.uint64(0)
        tmp = G
        G = G2
        G2 = tmp
    for a in range(2 * N):
        for b in range(W):
            genomes_out[a, b] = G[a, b]


@njit(parallel=True, cache=True)
def evolve_many(N, n_gen, WA, WX, posA, posX, RA, RX, UA, UX, eff_f, eff_m, noise_sd,
                male_recomb, seeds, out):
    for r in prange(seeds.shape[0]):
        evolve_one(N, n_gen, WA, WX, posA, posX, RA, RX, UA, UX, eff_f, eff_m, noise_sd,
                   male_recomb, seeds[r], out[r])


# ---------------------------------------------------------------------------
# measurement helpers (operate on final genomes; counting only)
# ---------------------------------------------------------------------------
@njit(cache=True)
def genotype_bv(genomes, idx, is_male, WA, WX, eff):
    """Breeding value of eff for individuals idx given their sex flags."""
    out = np.zeros(idx.shape[0])
    for k in range(idx.shape[0]):
        i = idx[k]
        g0 = genomes[2 * i]
        g1 = genomes[2 * i + 1]
        if is_male[k]:
            out[k] = _bv(g0, 0, WA + WX, eff) + _bv(g1, 0, WA, eff)
        else:
            out[k] = _bv(g0, 0, WA + WX, eff) + _bv(g1, 0, WA + WX, eff)
    return out


@njit(cache=True)
def unpack(genomes, rows, W):
    """Bit matrix [len(rows), W*64] (uint8) of selected haplotypes."""
    out = np.zeros((rows.shape[0], W * 64), np.uint8)
    for k in range(rows.shape[0]):
        g = genomes[rows[k]]
        for wi in range(W):
            w = g[wi]
            while w != np.uint64(0):
                b = _ctz(w)
                out[k, wi * 64 + b] = 1
                w &= w - np.uint64(1)
    return out

"""OpenCL (AMD GPU) backend for the individual-based simulators.

Same zero-leakage contract and the same return format as `kernels.py`; only
the execution substrate differs. Two kernels per generation:

  survive   (one work-item per individual): fair-coin sex, Bernoulli survival
            with probability weight / max weight in that sex; *measurement*
            of the generation's zygotes and surviving adults (atomic counts).
  reproduce (one work-item per offspring): mother and father drawn uniformly
            among surviving females / males by rejection sampling (pick a
            random individual until it is a surviving female) -- exactly
            uniform over survivors, no list compaction needed; Mendelian
            segregation (+ crossover with prob. r, or maternal cytoplasm).

Random numbers come from a counter-based hash (splitmix64 finaliser) keyed by
(seed, replicate, generation, individual, draw), so a run is reproducible and
independent of work-group scheduling. Results are statistically -- not
bit-for-bit -- equivalent to the CPU kernels (different random streams);
`tests/test_gpu.py` checks the equivalence.

Device: env SC_GPU_DEVICE (substring of the device name), default the first
discrete GPU found ('Baffin' = Radeon RX 560X on this machine).
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

try:
    import pyopencl as cl
except Exception:  # pragma: no cover - optional dependency
    cl = None

_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline ulong mix64(ulong z) {
    z += 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}
inline ulong key4(ulong seed, uint rep, uint gen, uint idx) {
    return mix64(mix64(mix64(seed + (ulong)rep) + (ulong)gen) + (ulong)idx);
}
/* uniform float in [0,1) from draw number s of a key */
inline float unif(ulong key, uint s) {
    return (float)(mix64(key + (ulong)s * 0x632BE59BD9B4E019UL) >> 40) * 5.9604644775390625e-8f;
}
/* uniform integer in [0,n) */
inline uint uidx(ulong key, uint s, uint n) {
    return mul_hi((uint)(mix64(key + (ulong)s * 0x632BE59BD9B4E019UL) >> 32), n);
}

/* ------------------------------------------------------------------ */
/* haploid two-locus (poster)  cls = 2*A + M                           */
/* ------------------------------------------------------------------ */
__kernel void hap_survive(__global const uchar *A, __global const uchar *M,
                          __global uchar *status, __global int *nsurv,
                          __global const float *psurv,   /* [2][4] */
                          __global int *rec, uint N, uint G, uint gen, ulong seed)
{
    uint gid = get_global_id(0);
    uint rep = gid / N, i = gid % N;
    ulong k = key4(seed, rep, gen, i);
    uint sex = unif(k, 0) < 0.5f ? 0u : 1u;
    uint c = 2u * A[gid] + M[gid];
    uchar st = 0;
    if (unif(k, 1) < psurv[sex * 4u + c]) {
        st = (uchar)(sex + 1u);
        atomic_inc(&nsurv[rep * 2u + sex]);
    }
    status[gid] = st;
    atomic_inc(&rec[((ulong)rep * (G + 1u) + gen) * 4u + c]);   /* measurement */
}

__kernel void hap_reproduce(__global const uchar *A, __global const uchar *M,
                            __global const uchar *status, __global const int *nsurv,
                            __global uchar *A2, __global uchar *M2,
                            float r, uint N, uint gen, ulong seed)
{
    uint gid = get_global_id(0);
    uint rep = gid / N, j = gid % N;
    if (nsurv[rep * 2u] == 0 || nsurv[rep * 2u + 1u] == 0) {   /* population failed: freeze */
        A2[gid] = A[gid]; M2[gid] = M[gid]; return;
    }
    ulong k = key4(seed ^ 0xD1B54A32D192ED03UL, rep, gen, j);
    uint base = rep * N, s = 0, mom, dad;
    do { mom = base + uidx(k, s++, N); } while (status[mom] != 1);
    do { dad = base + uidx(k, s++, N); } while (status[dad] != 2);
    uint p = mom, q = dad;
    if (unif(k, s++) >= 0.5f) { p = dad; q = mom; }
    A2[gid] = A[p];
    M2[gid] = (unif(k, s++) < r) ? M[q] : M[p];
}

/* ------------------------------------------------------------------ */
/* diploid autosomal + maternal cytoplasm  cls = 3*mt + x1 + x2        */
/* rec row: [A in zygotes, mito+ zygotes, A in adult F, n F, A in adult M, n M] */
/* ------------------------------------------------------------------ */
__kernel void dip_survive(__global const uchar *x1, __global const uchar *x2,
                          __global const uchar *mt, __global uchar *status,
                          __global int *nsurv, __global const float *psurv, /* [2][6] */
                          __global int *rec, uint N, uint G, uint gen, ulong seed)
{
    uint gid = get_global_id(0);
    uint rep = gid / N, i = gid % N;
    ulong k = key4(seed, rep, gen, i);
    uint sex = unif(k, 0) < 0.5f ? 0u : 1u;
    uint nA = x1[gid] + x2[gid];
    uint c = 3u * mt[gid] + nA;
    __global int *row = rec + ((ulong)rep * (G + 1u) + gen) * 6u;
    if (nA) atomic_add(&row[0], (int)nA);
    if (mt[gid]) atomic_inc(&row[1]);
    uchar st = 0;
    if (unif(k, 1) < psurv[sex * 6u + c]) {
        st = (uchar)(sex + 1u);
        atomic_inc(&nsurv[rep * 2u + sex]);
        atomic_inc(&row[3u + 2u * sex]);
        if (nA) atomic_add(&row[2u + 2u * sex], (int)nA);
    }
    status[gid] = st;
}

__kernel void dip_reproduce(__global const uchar *x1, __global const uchar *x2,
                            __global const uchar *mt, __global const uchar *status,
                            __global const int *nsurv,
                            __global uchar *y1, __global uchar *y2, __global uchar *mt2,
                            uint N, uint gen, ulong seed)
{
    uint gid = get_global_id(0);
    uint rep = gid / N, j = gid % N;
    if (nsurv[rep * 2u] == 0 || nsurv[rep * 2u + 1u] == 0) {
        y1[gid] = x1[gid]; y2[gid] = x2[gid]; mt2[gid] = mt[gid]; return;
    }
    ulong k = key4(seed ^ 0xD1B54A32D192ED03UL, rep, gen, j);
    uint base = rep * N, s = 0, mom, dad;
    do { mom = base + uidx(k, s++, N); } while (status[mom] != 1);
    do { dad = base + uidx(k, s++, N); } while (status[dad] != 2);
    y1[gid] = (unif(k, s++) < 0.5f) ? x1[mom] : x2[mom];
    y2[gid] = (unif(k, s++) < 0.5f) ? x1[dad] : x2[dad];
    mt2[gid] = mt[mom];                     /* maternal transmission */
}
"""


def available():
    if cl is None:
        return False
    try:
        return _ctx() is not None
    except Exception:
        return False


@lru_cache(maxsize=None)
def _ctx():
    want = os.environ.get("SC_GPU_DEVICE", "").lower()
    devs = [d for p in cl.get_platforms() for d in p.get_devices(device_type=cl.device_type.GPU)]
    if not devs:
        return None
    if want:
        devs = [d for d in devs if want in d.name.lower()] or devs
    else:  # prefer a discrete GPU (more compute units)
        devs = sorted(devs, key=lambda d: -d.max_compute_units)
    dev = devs[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, _SRC).build()
    kern = {n: cl.Kernel(prg, n) for n in ("hap_survive", "hap_reproduce", "dip_survive", "dip_reproduce")}
    return ctx, q, kern, dev.name


def device_name():
    return _ctx()[3]


def _psurv(wtab):
    w = np.asarray(wtab, float)
    return (w / w.max(1, keepdims=True)).astype(np.float32).ravel()


def haploid_ensemble(init_counts, N, n_gen, wtab, r, n_rep, seed, stop_on_m_absorb=False):
    """GPU twin of kernels.haploid_ensemble -> counts[n_rep, n_gen+1, 4].
    (stop_on_m_absorb is accepted for API parity; the GPU simply runs on --
    an absorbed modifier stays absorbed.)"""
    ctx, q, prg, _ = _ctx()
    mf = cl.mem_flags
    init = np.asarray(init_counts, np.int64).reshape(n_rep, 4)
    cls = np.concatenate([np.repeat(np.arange(4, dtype=np.uint8), c) for c in init])
    A = (cls // 2).astype(np.uint8)
    M = (cls % 2).astype(np.uint8)
    tot = n_rep * N
    bufA = [cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=A), cl.Buffer(ctx, mf.READ_WRITE, tot)]
    bufM = [cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=M), cl.Buffer(ctx, mf.READ_WRITE, tot)]
    status = cl.Buffer(ctx, mf.READ_WRITE, tot)
    nsurv = cl.Buffer(ctx, mf.READ_WRITE, 8 * n_rep)
    ps = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_psurv(wtab))
    rec_h = np.zeros(n_rep * (n_gen + 1) * 4, np.int32)
    rec = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rec_h)
    zero = np.zeros(1, np.int32)
    ks, kr = prg["hap_survive"], prg["hap_reproduce"]
    sd = np.uint64(seed)
    cur = 0
    for g in range(n_gen + 1):
        cl.enqueue_fill_buffer(q, nsurv, zero, 0, 8 * n_rep)
        ks(q, (tot,), None, bufA[cur], bufM[cur], status, nsurv, ps, rec,
           np.uint32(N), np.uint32(n_gen), np.uint32(g), sd)
        if g < n_gen:
            kr(q, (tot,), None, bufA[cur], bufM[cur], status, nsurv, bufA[1 - cur], bufM[1 - cur],
               np.float32(r), np.uint32(N), np.uint32(g), sd)
            cur = 1 - cur
        if g % 256 == 255:
            q.finish()  # keep the command queue short
    cl.enqueue_copy(q, rec_h, rec)
    q.finish()
    return rec_h.reshape(n_rep, n_gen + 1, 4).astype(np.int64)


def diploid_ensemble(init_alleles, init_mito, N, n_gen, wtab, n_rep, seed):
    """GPU twin of kernels.diploid_ensemble -> rec[n_rep, n_gen+1, 6]."""
    ctx, q, prg, _ = _ctx()
    mf = cl.mem_flags
    rng = np.random.default_rng(seed)
    x1 = np.empty((n_rep, N), np.uint8)
    x2 = np.empty((n_rep, N), np.uint8)
    mt = np.empty((n_rep, N), np.uint8)
    for rep in range(n_rep):  # founders: copies placed at random on chromosomes
        genes = np.zeros(2 * N, np.uint8)
        genes[:init_alleles] = 1
        rng.shuffle(genes)
        x1[rep], x2[rep] = genes[:N], genes[N:]
        m = np.zeros(N, np.uint8)
        m[:init_mito] = 1
        rng.shuffle(m)
        mt[rep] = m
    tot = n_rep * N
    B = lambda a: cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a).ravel())
    b1, b2, bm = [B(x1), cl.Buffer(ctx, mf.READ_WRITE, tot)], [B(x2), cl.Buffer(ctx, mf.READ_WRITE, tot)], \
                 [B(mt), cl.Buffer(ctx, mf.READ_WRITE, tot)]
    status = cl.Buffer(ctx, mf.READ_WRITE, tot)
    nsurv = cl.Buffer(ctx, mf.READ_WRITE, 8 * n_rep)
    ps = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=_psurv(wtab))
    rec_h = np.zeros(n_rep * (n_gen + 1) * 6, np.int32)
    rec = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rec_h)
    zero = np.zeros(1, np.int32)
    ks, kr = prg["dip_survive"], prg["dip_reproduce"]
    sd = np.uint64(seed)
    cur = 0
    for g in range(n_gen + 1):
        cl.enqueue_fill_buffer(q, nsurv, zero, 0, 8 * n_rep)
        ks(q, (tot,), None, b1[cur], b2[cur], bm[cur], status, nsurv, ps, rec,
           np.uint32(N), np.uint32(n_gen), np.uint32(g), sd)
        if g < n_gen:
            kr(q, (tot,), None, b1[cur], b2[cur], bm[cur], status, nsurv, b1[1 - cur], b2[1 - cur], bm[1 - cur],
               np.uint32(N), np.uint32(g), sd)
            cur = 1 - cur
        if g % 256 == 255:
            q.finish()
    cl.enqueue_copy(q, rec_h, rec)
    q.finish()
    return rec_h.reshape(n_rep, n_gen + 1, 6).astype(np.int64)

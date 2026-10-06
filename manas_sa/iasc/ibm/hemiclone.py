"""Hemiclonal assay emulator (Geeta Arun et al. 2022, Methods), applied to a
simulated LH-like population.

Genetics of the assay (clone-generator design)
----------------------------------------------
* a hemigenome h = one haploid genome (autosome + X) sampled from the population;
* focal female = h + a random population haplotype b (X heterozygous);
* focal male   = autosome h + autosome b; X = h's X, hemizygous (sons of a
  hemiclone sire x compound-X DxLH dam inherit the sire's X);
* competitors (LHst) = random population individuals.

Phenotypes (one adult generation, sex-ratio treatment for 2 days)
-----------------------------------------------------------------
Sex ratio enters only through a per-sex selection-intensity multiplier
beta_s(SR) on the breeding value (BV) during the assay:
  female: eggs of each focal female ~ Poisson(lam0 * exp(beta_f(SR)*BV_f + e_ind + e_vial + e_day))
          vial fitness = eggs of the 2 focal females (as measured)
  male:   each scored female's offspring split between focal and competitor
          males with focal share = sum_focal exp(beta_m*BV + e) / sum_all(...),
          offspring ~ Poisson, focal-sired ~ Binomial; vial fitness =
          proportion sired by focal males over 7 scored females (as measured)
beta and the noise sds are calibrated only to line VARIANCES (thesis Ch. 4);
r_w,g,mf, its sex-ratio ordering, the SA proportion and the across-sex-ratio
correlations are predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine as E

SR = ("M", "E", "F")


def sample_hemigenomes(genomes, arch, rng, n_lines):
    """Clone-generator protocol: each hemigenome is the haploid genome of a
    sperm of a population MALE: his X (maternal slot 2i; the paternal slot
    carries the Y) plus a random one of his two autosome sets.
    Returns an (n_lines, WA + WX) haplotype array."""
    N = genomes.shape[0] // 2
    WA = arch.WA
    nf_ = N // 2
    males = nf_ + rng.choice(N - nf_, n_lines, replace=n_lines > N - nf_)
    hapA = 2 * males + rng.integers(0, 2, n_lines)
    H = genomes[hapA].copy()
    H[:, WA:] = genomes[2 * males, WA:]
    return H


def hemigenome_bv(genomes, arch, rng, n_lines, n_background):
    """BVs needed for the assay. Returns dict with arrays:
    h_f[l], h_mA[l] (autosome only), h_mX[l]; bg_f[k], bg_mA[k];
    comp_f[k], comp_m[k] (whole competitor individuals)."""
    return assay_bv(genomes, arch, rng, sample_hemigenomes(genomes, arch, rng, n_lines), n_background)


def assay_bv(genomes, arch, rng, H, n_background):
    """Assay BVs for given hemigenome haplotypes H (rows = lines), with focal
    backgrounds and competitors drawn from the population `genomes`."""
    N = genomes.shape[0] // 2
    WA, WX = arch.WA, arch.WX
    nf_ = N // 2
    # the other haplotype of a focal individual comes from a random population
    # female (both of her slots carry an autosome and an X)
    bgA = rng.integers(0, 2 * nf_, n_background)
    bgX = rng.integers(0, 2 * nf_, n_background)

    def hap_bv(src, rows, eff, w0, w1):
        out = np.empty(len(rows))
        for k, r in enumerate(rows):
            out[k] = E._bv(src[r], w0, w1, eff)
        return out
    lines = np.arange(len(H))
    d = dict(
        h_f=hap_bv(H, lines, arch.eff_f, 0, WA + WX),
        h_mA=hap_bv(H, lines, arch.eff_m, 0, WA),
        h_mX=hap_bv(H, lines, arch.eff_m, WA, WA + WX),
        bg_f=hap_bv(genomes, bgA, arch.eff_f, 0, WA) + hap_bv(genomes, bgX, arch.eff_f, WA, WA + WX),
        bg_mA=hap_bv(genomes, bgA, arch.eff_m, 0, WA),
    )
    # competitors: random whole individuals (sex-appropriate genotype)
    nf = N // 2
    fi = rng.integers(0, nf, n_background)
    mi = nf + rng.integers(0, N - nf, n_background)
    d["comp_f"] = E.genotype_bv(genomes, fi, np.zeros(len(fi), np.bool_), WA, WX, arch.eff_f)
    d["comp_m"] = E.genotype_bv(genomes, mi, np.ones(len(mi), np.bool_), WA, WX, arch.eff_m)
    # standardise on the population scale of whole-individual breeding values so
    # that beta (calibrated) is "sd of log fitness per sd of breeding value" and
    # does not depend on the rescaling factor lambda
    sf = d["comp_f"].std() or 1.0
    sm = d["comp_m"].std() or 1.0
    for k in ("h_f", "bg_f", "comp_f"):
        d[k] = d[k] / sf
    for k in ("h_mA", "h_mX", "bg_mA", "comp_m"):
        d[k] = d[k] / sm
    return d


def run_assay(bv, acfg, beta, noise, rng, sexes=("Female", "Male"), srs=SR):
    """Vectorised assay. Returns vial-level DataFrame with the authors' columns
    (Sex, SexRatio, Day, Line, Fitness).
    beta = {(sex, SR): multiplier on BV}; noise = {(sex, SR): sd} (log scale,
    vial-level for females; per-male competitive noise for males) plus
    noise['day'] (day effect sd, both sexes)."""
    L = len(bv["h_f"])
    nb = len(bv["bg_f"])
    D = acfg["days_per_sex_ratio"]
    mean_f = bv["comp_f"].mean()
    mean_m = bv["comp_m"].mean()
    frames = []
    for sr in srs:
        if "Female" in sexes:
            V = acfg["female_vials_per_day"]
            g = bv["h_f"][None, :, None, None] + bv["bg_f"][rng.integers(nb, size=(D, L, V, 2))] - mean_f
            e = rng.normal(0, noise[("Female", sr)], (D, L, V, 1)) + rng.normal(0, noise["day"], (D, 1, 1, 1))
            lam = acfg["eggs_per_female_mean"] * np.exp(beta[("Female", sr)] * g + e)
            eggs = rng.poisson(lam).sum(-1).astype(float)
            dd, ll, vv = np.meshgrid(np.arange(D), np.arange(L), np.arange(V), indexing="ij")
            frames.append(pd.DataFrame({"Sex": "Female", "SexRatio": sr, "Day": [f"F{sr}{x}" for x in dd.ravel()],
                                        "Line": ll.ravel() + 1, "Fitness": eggs.ravel()}))
        if "Male" in sexes:
            V = acfg["male_vials_per_day"]
            nM = acfg["males_per_vial"][sr]
            nfoc = max(1, nM // 4)
            ncomp = nM - nfoc
            gf = (bv["h_mA"] + bv["h_mX"])[None, :, None, None] + bv["bg_mA"][rng.integers(nb, size=(D, L, V, nfoc))] - mean_m
            gc = bv["comp_m"][rng.integers(nb, size=(D, L, V, ncomp))] - mean_m
            sd = noise[("Male", sr)]
            day = rng.normal(0, noise["day"], (D, 1, 1, 1))
            wf = np.exp(beta[("Male", sr)] * gf + rng.normal(0, sd, gf.shape) + day)
            wc = np.exp(beta[("Male", sr)] * gc + rng.normal(0, sd, gc.shape))
            share = wf.sum(-1) / (wf.sum(-1) + wc.sum(-1))                      # (D, L, V)
            nfem = acfg["females_scored_per_male_vial"]
            n_off = rng.poisson(acfg["offspring_per_female"], (D, L, V, nfem))
            foc = rng.binomial(n_off, np.broadcast_to(share[..., None], n_off.shape))
            fit = foc.sum(-1) / np.maximum(n_off.sum(-1), 1)
            dd, ll, vv = np.meshgrid(np.arange(D), np.arange(L), np.arange(V), indexing="ij")
            frames.append(pd.DataFrame({"Sex": "Male", "SexRatio": sr, "Day": [f"M{sr}{x}" for x in dd.ravel()],
                                        "Line": ll.ravel() + 1, "Fitness": fit.ravel()}))
    return pd.concat(frames, ignore_index=True)

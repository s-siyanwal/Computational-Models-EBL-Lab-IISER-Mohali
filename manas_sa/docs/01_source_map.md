# 1. Source map: Manas Geeta Arun (M. A. Samant) sexual-antagonism results

Every row is tagged with its claim type:

- **A** = reported in the paper.
- **B** = reproduced by this repository.
- **C** = our extrapolation.

"Recomputed" means we re-ran the paper's estimator on the authors' own deposited data.

## Access status

| # | Target | Status | What we used |
|---|---|---|---|
| 1 | Geeta Arun *et al.* 2022, *BMC Ecol Evol* 22:38 | **Full text + raw data** | `experimental_papers_manas/s12862-022-01992-0.pdf`; raw data = Additional file 2 (`12862_2022_1992_MOESM3_ESM.xlsx`, 2,539 vial rows); sample sizes = Additional file 1 |
| 2 | Samant 2022 PhD thesis (IISER Mohali) | **Full text** | `experimental_papers_manas/Thesis.pdf`. Ch. 3 = target 1. Ch. 4 = interlocus-conflict traits and their cross-sex genetic correlations. Ch. 6 = the two-locus *haploid* modifier model (the IISER poster). |
| 3 | Geeta Arun *et al.* 2025, *Am Nat* 206(6):552–568 (Robertson covariances) | **NOT ACCESSIBLE** | The journal returned HTTP 403 and no preprint was found. We use **thesis Ch. 4** (same hemigenome panel and sex-ratio design, trait–fitness genetic correlations) as a marked substitute. **No Am Nat number is quoted anywhere in this repository.** |
| 4 | Geeta Arun 2025, bioRxiv 10.1101/2025.01.08.632022 ("Tie that binds") | **Full text** (downloaded from bioRxiv) | `experimental_papers_manas/downloaded/GeetaArun2025_HRI_biorxiv_2025.01.08.632022v1.pdf` |
| 5 | Singh, Jain, Geeta Arun & Prasad 2023, bioRxiv 10.1101/2023.10.25.564090 | **Full text + supplement with the authors' Python code** | Two identical copies of the PDF; `media-1.docx` = supplementary information with recursions and 7 code listings; `media-2.mp4` = a supplementary video |

**Wrong-paper note.** `experimental_papers_manas/2022.08.04.502812v1.full.pdf` is Singh, Hasan & Agrawal (2023, *Evolution* 77:2015), a DSPR study from Toronto. It is **not** the Am Nat target, and it is not by Manas.

The context papers present are Syed *et al.* 2017 and 2020, and Geeta Arun *et al.* 2021 (*Evol Lett*, sex-specific dominance for infection survival). Not in the folder: Nandy *et al.* 2013/2014 and Maggu *et al.* 2021. Those are cited only via the thesis.

## Target 1: hemiclonal analysis at three adult sex ratios (A unless marked)

**Design**
- 39 hemigenomes from the LH population (43 sampled; 4 lost).
- Each was expressed in both sexes on random LH backgrounds, via clone generators (compound X plus a T(2;3) translocation). The dot chromosome is excluded.
- Adult sex ratio was imposed on days 12–14 only (2 days of a 14-day cycle), always 32 flies per vial:
  - male-biased (MB): 24♂:8♀
  - equal (E): 16♂:16♀
  - female-biased (FB): 8♂:24♀
- Female fitness = eggs laid in 18 h by 2 focal females after competition for limiting yeast. Yeast was adjusted to 0.47 mg per female, focal:competitor = 1:3, and 7 vials × 2 days per line × sex ratio.
- Male fitness = proportion of progeny sired in competition with LHst males, with 5 vials × 7 females × 2 days.
- 2 replicate days per sex ratio.

**Estimators**
- **(a)** Line averages. Male fitness is arcsine-square-root transformed and each day is divided by its mean. Then the mean of the per-day line means is taken, and values are scaled and centred per sex × sex ratio.
  - r_w,g,mf = Pearson correlation of line means.
  - The proportion of sexually antagonistic (SA) variation comes from a 45° rotation.
  - Bootstrap: 10,000 resamples stratified by sex × line × day (CI type not stated).
- **(b)** MCMCglmm with sex-specific line effects plus a day × line term. Heritability h² = 2σ²_line/(σ²_line + σ²_res); r_mf = σ_mf/(σ_m σ_f).

**Results** (95% CI in brackets)

| Statistic | Male-biased | Equal | Female-biased | Different from 0? |
|---|---|---|---|---|
| r_w,g,mf (line means), Table 2A | 0.3805 [0.2992, 0.5283] | 0.4027 [0.3140, 0.5526] | 0.2515 [0.1198, 0.4502] | all > 0 |
| r_w,g,mf, MB − FB difference | 0.129 (CI [−0.0721, 0.3507]) | | | **n.s.** |
| SA proportion of fitness variance, Table 2A | 0.3097 [0.2358, 0.3504] | 0.2986 [0.2237, 0.3430] | 0.3742 [0.2749, 0.4401] | MB − FB difference CI [−0.1753, 0.0360]: **n.s.** |
| r_w,g,mf (MCMCglmm), Table 2B | 0.5056 [0.1418, 0.7983] | 0.4999 [0.1397, 0.7787] | 0.4462 [0.0059, 0.8470] | all > 0; difference n.s. |
| h² females, Table 2B | 0.8702 [0.5935, 1.1520] | 0.9992 [0.7337, 1.2696] | 0.7385 [0.5021, 1.0539] | |
| h² males, Table 2B | 0.4788 [0.2383, 0.7303] | 0.5762 [0.3192, 0.8637] | 0.2229 [0.0495, 0.4080] | female > male is significant at MB and E |
| r between sex ratios, females (line means) | MB–FB 0.7688 | MB–E 0.7493 | FB–E 0.8421 | |
| r between sex ratios, males (line means) | MB–FB 0.5567 | MB–E 0.6995 | FB–E 0.5415 | |
| LMM likelihood-ratio tests | line p = 0.0237 | line × sex p < 0.0001 | line × sex ratio p = 0.82 | line × sex × sex ratio p = 0.0002 |

**Authors' mechanistic claim**
- Stronger interlocus sexual conflict (IeSC, at MB) *non-significantly* weakens intralocus conflict (IaSC).
- r is significantly positive, unlike earlier LHM estimates.
- Possible mechanisms they discuss: (i) selection on shared traits such as locomotion turns concordant under intense IeSC; (ii) positive LD/genetic correlation between persistence and resistance.

**Flags**
- **F1.** In Table 2B, the male "MB–FB" between-sex-ratio row (0.8932 [0.6888, 0.9996]) is identical to the female row. This is probably a copy error.
- **F2.** Several Table 2A bootstrap CIs barely contain their own estimate (female FB–E: 0.8421 with [0.8403, 0.8956]). The CI method is unspecified.
- **F3.** The README says "six days", but the data contains labels Day 1–Day 12 (separate days per sex). This is harmless.
- **F12.** The Methods say male fitness was arcsine-square-root transformed before line means were taken. But all 12 Table 2A line-mean statistics are reproduced to the 4th decimal only *without* the transform. With the transform, r(E) = 0.419 and the male across-SR correlations are about 0.03 higher. The transform was presumably applied only in the mixed models.

**B (recomputed from the raw data, `analysis/recompute_bmc2022.py`, 10,000 stratified bootstrap resamples)**
- All 12 Table 2A line-mean point estimates are reproduced exactly (to 4 decimals):
  - r = 0.3805 / 0.4027 / 0.2515;
  - SA proportion = 0.3097 / 0.2986 / 0.3742;
  - all six across-SR correlations;
  - MB − FB difference = 0.129.
  This requires untransformed male proportions (F12).
- The MB − FB difference CI is [−0.090, 0.330], against the published [−0.072, 0.351]. Both include 0.
- The CI construction is not identified (F2). Resampling vials within cells attenuates correlations, so percentile intervals sit *below* the estimates. The lower bounds of the "basic" (reflected) interval reproduce the published lower bounds within about 0.01: 0.308 vs 0.299, 0.317 vs 0.314, 0.125 vs 0.120, 0.745 vs 0.744. The upper bounds do not.

## Target 2: thesis (only what targets 1, 3 and 5 lack)

**Ch. 3.** Identical numbers to target 1 (A).

**Ch. 4, the substitute for the Am Nat paper (A).** Same hemigenome panel.
- **Line variance of relative fitness** (line mean ÷ mean):

  | Sex | MB | E | FB | Levene test |
  |---|---|---|---|---|
  | Males | 0.2926 [0.1954, 0.4859] | 0.1591 [0.1062, 0.2642] | 0.1010 [0.0674, 0.1678] | p = 0.0174 |
  | Females | 0.0532 | 0.0321 | 0.0254 | p = 0.39 |

- LHst-female fecundity was lower at MB than at FB (p < 0.0001), i.e. male harm.
- Intersexual genetic correlations for interlocus-conflict traits are all n.s.:
  - latency to first mating −0.018 [−0.544, 0.522]
  - copulation duration 0.282 [−0.452, 0.977]
  - remating latency 0.002
  - mating rate MB 0.259, FB −0.039
- r(P1, P2) = 0.787 [0.528, 0.998].
- **Cross-sex trait–fitness correlations** (Table 4.6A, male traits vs female fitness; FB / E / MB):
  - latency to first mating −0.466 (p = 0.003) / −0.511 (p = 0.0009) / −0.262 (p = 0.11)
  - P1 0.362 / 0.306 / 0.177
  - P2 0.423 / 0.314 / 0.265

  Male persistence is *positively* genetically correlated with female fitness, and these cross-sex correlations weaken at MB.
- Table 4.6B: female mating rate vs male fitness at FB = −0.342 (p = 0.039).
- Within-sex linear selection gradients on male traits are steepest at MB. The MB column of Tables 4.4/4.5 is garbled in the text layer, so we use no MB gradient values (flag F4).

**Ch. 6.** The haploid two-locus modifier model (the poster). It was reconstructed and verified in `../results/REPORT.md` of the parent project (thresholds within about 0.002 of the poster).

## Target 4: "Tie that binds", HRI preprint (A)

**Model**
- SLiM 4.2, Wright–Fisher, one autosome of 1 Mb.
- N = 2,500 (fixed), 25,000 generations, starting from blank genomes.
- Two mutation types, female-only and male-only, both deleterious and additive (h = 0.5).
- s ~ reflected gamma (shape 0.3, scale 0.05).
- Breeding value = Σ s over carried mutations of the own sex's type, plus N(0,1) noise; absolute fitness = exp(relative fitness).
- Recombination uses SLiM defaults, i.e. **both sexes recombine** (flag F5: unlike real *D. melanogaster* males).

**Estimator**
- From a sample of 250 ♀ + 250 ♂: V_f = α_fᵀ L α_f, V_m = α_mᵀ L α_m and COV = α_fᵀ L α_m.
- L is the genotype (co)variance matrix: diagonal p(1−p)/2, off-diagonal D_ij.
- r_mf,W = COV/√(V_f V_m). Selective interference = COV/V_f × 100.

**Results**

| Quantity | Value |
|---|---|
| Mean r_mf,W, map length 0.1 M and μ = 7×10⁻⁷ | **−0.0780**; replicates range about −0.300 to 0.125 (50 replicates) |
| Bias vs map length (0.001–10 M) | stronger at low recombination, **absent at 10 M** (Fig. 2: boxplots only, no numbers) |
| Bias vs mutation rate (5×10⁻⁷ to 1.3×10⁻⁶) | stronger at higher μ (Fig. 3, boxplots only) |
| V_f, V_m | about 0.02 |
| Segregating sites | about 8,000, rising to over 13,000 at 10 M |
| Median interference | about 10% at low map length, falling to about 0 at 10 M; 5% → just under 15% with μ |

**Mechanistic claim**
- COV_mf,W = Σ_i (p_i q_i/2) α_f,i α_m,i (direct) + Σ_{i≠j} D_ij α_f,i α_m,j (indirect).
- With purely sex-limited selection the direct term is 0.
- HRI makes D < 0 between linked beneficial alleles, so COV < 0 without any SA selection.

## Target 5: imperfect modifiers, diploid two-locus model (A)

**Model**
- Connallon & Clark (2010) recursions (diploid, autosomal or X-linked, sex-specific recombination r_m and r_f).
- Modifier B₂ reduces the female expression of A₁ (factor k₁) and of A₂ (factor k₂).
- Dominance reversal with h_m = h_f = h ∈ (0, 0.5].
- Fitness tables 1A/1B (autosomal) and 2A/2B (X-linked).
- Resolution = A₂B₂ fixed within 3,000 generations.

**From the authors' code** (supplement, `PSAcommented.py` and `PSXcommented.py`)
- s_f ∈ {0.001, 0.2, …, 1.0}; h ∈ {0.1, 0.2, 0.3, 0.4, 0.49}; r_m, r_f ∈ {0.1, …, 0.4}.
- s_m spans the Connallon–Clark polymorphism window in steps of 0.1.
- B₂ starts at 0.05 and A₂ starts at its single-locus equilibrium.
- D sweeps a range bounded by haplotype frequencies (in practice a few hundredths).
- "Fixed" means the frequency rounds to 1.000.

**Results**
- k₁ = 0, k₂ = 1: A₂B₂ fixed in **95.7%** of parameter space.
- k₂ = 1: more than **85%** for any k₁.
- Resolution increases with k₂ and is non-monotonic in k₁ (hardest at intermediate k₁).
- The X is more conducive than autosomes overall, *except* at very high k₂, where autosomes win.
- Increasing h hinders resolution.
- Higher s_m relative to s_f favours resolution.
- Recombination has no effect unless r_m = r_f = 0, where resolution is lower.

**Flags**
- **F6.** Table 3 gives D ∈ [−0.25, 0.25], but the text and code use about ±0.02.
- **F7.** The figure captions say "k₁ step-size = 1"; it is 0.1.
- **F8.** Table 3 gives r ∈ [0, 0.5], but the code uses 0.1–0.4.
- **F9.** The autosomal code's upper s_m bound `sf*(1-h)/h*(1-sf)` has an operator-precedence slip.
- **F10.** F₂₂r and F₂₃ use absolute values.
- **F11.** Figures 1–2 are heatmaps with no deposited numbers. We can score only the two quoted percentages and the qualitative orderings.

## Contrast set (A; values from primary texts)

| Study | Population, number of lines | Fitness measures | r_mf for fitness |
|---|---|---|---|
| Chippindale, Gibson & Rice 2001, PNAS (PMC29315) | LHM, 40 genomes | ♂ % females mated (deviation); ♀ per-capita fecundity | adult **−0.30** (P = 0.03); juvenile +0.49; total −0.16 (n.s.) |
| Innocenti & Morrow 2010, PLoS Biol | LHM, 100 | competitive fertilization; competitive fecundity | **−0.52** [−0.86, −0.10] |
| Collet *et al.* 2016, Evolution (PMC5069644) | LHM-UU, 100 / LHM-UCL, 113 | as above | **−0.41** (SE 0.18) / **+0.21** (SE 0.19); difference p = 0.02 |
| Ruzicka *et al.* 2019 (preprint 117176) | LHM, 223 | lifetime reproductive success (LHM regime) | **+0.15** [−0.21, 0.46]; h²: ♀ 0.42, ♂ 0.16 |
| Geeta Arun *et al.* 2022 | LH (descends from LHM; >500 generations), 39 | as target 1 | **+0.25 to +0.40** (line means) |

## Coverage of reported statistics

The full table is in `results/coverage.csv`. The denominators are the statistics this repository attempts:

| Source | Scoreable | Not scoreable |
|---|---|---|
| Target 1 | 21 numeric statistics | Daily random-effect variances, and MCMCglmm-specific heritabilities beyond r |
| Target 2 | Ch. 4 line variances and cross-sex correlations | Trait data are not deposited, so trait-level Robertson covariances can't be recomputed |
| Target 3 | 0 | All; inaccessible |
| Target 4 | Mean r at 0.1 M, the sign and trend vs recombination and μ, V about 0.02, interference about 10% | Per-map-length values: boxplots only |
| Target 5 | The 95.7% and >85% values and qualitative orderings | Heatmap cell values |

# 2. Model specification

There are three nested layers, which share one configuration file (`config/default.toml`):

1. an **analytical core** of exact recursions;
2. an **individual-based model (IBM)**: a numba forward simulator and hemiclone-assay emulator, with SLiM 5.2 equivalents;
3. a thin **emulator**, used for design and sensitivity only.

## 2.1 What is sexually antagonistic vs sex-limited

Each mutation (site) carries a pair of per-copy effects on relative fitness, *(e_f, e_m)*. Every class draws its magnitude s from Gamma(shape, scale); the scale is rescaled in Layer 2.

| Kind | e_f | e_m | Class |
|---|---|---|---|
| 1 | −s | 0 | female-limited |
| 2 | 0 | −s | male-limited |
| 3 | −s | −s | sexually concordant (SC) |
| 4 | −s | +b·s | sexually antagonistic (SA), harms females |
| 5 | +b·s | −s | SA, harms males |

b = `sa_benefit_ratio` = 0.5 [ours]. With b = 1 an additive SA allele is exactly neutral when averaged over the sexes. At the genomic mutation rate used here (U ≈ 0.9 per genome per generation) such alleles accumulate without bound: in pilot runs V_g reached the thousands and r went to −1. With b < 1 they are net deleterious and reach mutation–selection–drift balance. Balancing selection through dominance reversal is not modelled in the IBM; it is treated analytically in target 5.

There are two regimes:

- **(i) True SA selection:** kinds 4 and 5 present.
- **(ii) Purely sex-limited selection:** kinds 1 and 2 only. No locus affects both sexes, so the *direct* term of COV_mf,W is exactly 0.

Interlocus conflict (persistence/resistance traits) is represented *phenomenologically*. Adult sex ratio rescales how strongly each sex's breeding value maps onto that sex's measured fitness component (§2.4). Explicit harm and resistance traits are not modelled; claim type C notes where this limits inference.

## 2.2 Layer 1: analytical core (`iasc/analytic/`)

**Covariance decomposition** (Geeta Arun 2025, eq. 1):

> COV_mf,W = α_fᵀ L α_m = Σ_i (p_i q_i/2) α_f,i α_m,i + Σ_{i≠j} D_ij α_f,i α_m,j

The first sum is the direct term and the second the indirect term. The code computes the same quantity as Cov(BV_f, BV_m) over genotypes.

**Exact two-sex, two-locus recursion** (`twolocus.py`)
- Diploid; autosomal or X-linked.
- Arbitrary 4×4 genotype fitness tables per sex.
- Sex-specific crossover rates r_f and r_m; Drosophila has r_m = 0.
- Optional recurrent mutation.
- It generalises Connallon & Clark (2010) and reduces to it.

**Derived result** (claim type C, verified against the exact recursion). Purely sex-limited selection makes the egg and sperm pools differ in allele frequency: Δ_A at a female-limited locus, Δ_B at a male-limited one. Gamete formation then creates

> D′ = ¼ (1 − 2r) Δ_A Δ_B,

which is negative for female-beneficial × male-beneficial alleles. At steady state,

> D* ≈ (1 − 2r̄) Δ_A Δ_B / (4 r̄),  with r̄ = (r_f + r_m)/2.

With r_m = 0 this term persists even between loci at opposite ends of a chromosome (r_f = ½); it vanishes only between chromosomes. It is deterministic, so it exists at infinite N. That is a second route, alongside Hill–Robertson interference (HRI), to a negative indirect term.

**Modifier models** (`modifier_cc.py`)
- An exact, compiled re-implementation of the diploid imperfect-modifier model (target 5), using the authors' own code logic.
- The haploid poster model (thesis Ch. 6) is reused from the parent project (`../sexconflict/deterministic/poster_model.py`).

## 2.3 Layer 2: individual-based simulator (`iasc/ibm/engine.py`)

**Life cycle per generation** (Wright–Fisher with sexes, as in SLiM WF mode)
1. Compute breeding values: BV_f = Σ e_f over carried derived copies, and BV_m similarly. A male's X counts once (hemizygous).
2. Absolute fitness = exp(BV + N(0, 1)), the preprint's model.
3. Each of the N offspring draws a mother ∝ female absolute fitness and a father ∝ male absolute fitness. Offspring are N/2 daughters and N/2 sons.
4. Meiosis: Poisson(R) crossovers at uniform map positions in the mother (autosome and X). The father contributes an un-recombined autosome unless `male_recombination = true`, plus his X to daughters or a Y to sons.
5. Mutation: Poisson(U) new derived copies per gamete at uniform finite sites (32,768 autosomal and 8,192 X-linked bitset sites).

**Validated against theory** (`tests/test_core.py`, 8 tests, all passing)
- Neutral π = 4N_eU: N_e/N = 0.977, expected 1.
- With σ = 1 lognormal noise, N_e/N is within 2% of the exact finite-N expectation (1/N_f)/E[Σp_i²] = 0.399 for 75 parents per sex. The large-N limit e⁻¹ = 0.368 is *not* the right target at that N.
- Crossover fractions equal Haldane's (1 − e^(−2d))/2 to three decimals.
- Female-only recombination doubles LD at a given map distance.

**Statistics** (`stats.py`). r_mf,W, V_f, V_m, COV, direct, indirect and interference = COV/V_f × 100, all computed on sampled genotypes exactly as in the preprint. Signed LD between female-limited and male-limited derived alleles is reported against map distance.

## 2.4 Hemiclone assay emulator (`iasc/ibm/hemiclone.py`) and the sex-ratio mapping

**Assay genetics**
- A hemigenome is the haploid genome of one sperm of a population **male**: his X (necessarily maternal) plus a random one of his two autosome sets. This follows the clone-generator protocol. An earlier version tied the autosome to the X slot and so never sampled paternal autosomes; this is fixed.
- Focal female = hemigenome + a random haplotype (heterozygous X).
- Focal male = hemigenome autosome + a random autosome, plus the hemigenome X *hemizygous*.
- Competitors are random population members, at a focal:competitor ratio of 1:3.
- Vial numbers, days and scored females follow the paper exactly.

**Measured fitness components (not a generic scalar)**
- Females: eggs laid by 2 focal females, eggs ~ Poisson(λ₀ exp(β_f(SR)·BV_f + vial + day noise)).
- Males: proportion of progeny sired among 7 scored females. The focal share is Σ_focal exp(β_m BV + ε) / Σ_all exp(·), and progeny are binomial.

**Sex ratio → selection.** Sex ratio enters only as a per-sex multiplier β_s(SR) on the breeding value during the 2-day treatment:
- MB = 24♂:8♀, E = 16:16, FB = 8:24.
- More males per female means more intense male–male competition and more male harm to females.

β and the noise sd are *calibrated to variances only*: the line variance of relative fitness (thesis Ch. 4) and the within-line×day CV² of vial fitness, both computed from the raw data. Everything else is out of sample:
- r_w,g,mf and its ordering across sex ratios
- the SA proportion
- the across-sex-ratio correlations
- the cross-/within-sex covariance ratios

These are produced by the authors' estimator code (`iasc/estimators.py`), which is the same code that recomputes the empirical values from their raw data.

**Flag F12 (scoring rule).** The Methods state that male fitness was arcsine-square-root transformed. Yet every Table 2A line-mean statistic (r_w,g,mf, SA proportion, all six across-sex-ratio correlations) is reproduced to the 4th decimal only **without** the transform. With it, r = 0.382 / 0.419 / 0.236 and the male across-SR correlations are 0.03 higher. Simulated assays are therefore scored without the transform (`ARCSINE_MALES = False`), exactly as the published numbers were computed.

## 2.5 Layer 3: emulator (`iasc/emulator/`, `scripts/run_design.py`, `scripts/run_emulator.py`)

- **Model:** a Gaussian-process regressor (constant × RBF + white noise, standardised inputs), with a gradient-boosting comparison.
- **Training data:** IBM outputs only. A Latin-hypercube design over the LH architecture:
  - f_SA ∈ [0, 0.5], the fraction of new mutations that are SA;
  - the concordant share of the non-SA mutations, ∈ [0, 1];
  - log₁₀ female autosome map length, ∈ [0.1, 2] M.

  Each design point has 3 replicate populations. Each population gets 20 calibrated 39-line assays, plus assays with 40, 100, 113 and 223 lines (the contrast studies' sizes).
- **Outputs:** the true hemigenome r and the 39-line assay r_w,g,mf (equal sex ratio).
- **Validation:** grouped cross-validation, with whole design points held out. RMSE is compared with the replicate-population noise floor.
- **Uses** (design / sensitivity / inversion only):
  1. first-order sensitivity indices;
  2. which f_SA makes the assay return each published r;
  3. whether a published value is within sampling reach of the LH-like architecture.

  No claim about Manas's results rests on the emulator alone; every direction it suggests is checked in the IBM.

## 2.6 Power and the distinguishing experiment (`scripts/run_design.py`, `iasc/ibm/experiments.py`, `scripts/run_distinguish.py`)

- **Power for the sex-ratio ordering.** The saved "shared" LH populations are assayed with 39, 80, 160 and 320 lines using the same calibrated instrument. Power = P(the 95% line-bootstrap CI of r(MB) − r(FB) excludes 0).
- **SA vs HRI experiment (in silico).** Hemigenomes sampled from a population are passed through k generations of *female* meiosis among themselves, with no selection or mutation. Map lengths are per real meiosis, i.e. not λ-rescaled. Then r between hemigenome female and male values is recomputed:
  - SA (pleiotropy, the direct term) predicts that r persists;
  - LD (HRI or sex-limited selection with linkage, the indirect term) predicts that r decays towards the linkage-equilibrium value r_LE, obtained by independently permuting every site.

  Power is computed for an experiment that assays n original and n recombinant lines with the calibrated instrument.

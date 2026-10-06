# 6. Chinmay (2019): behavioural and non-genetic extensions

**Target.** Temura Chinmay Krishna Yadav, *Studying male mate choice and non-genetic inheritance in laboratory-adapted populations of* Drosophila melanogaster *evolved for higher immunity against a gram-negative bacterium* Pseudomonas entomophila. BS–MS thesis, IISER Mohali, April 2019, supervisor N. G. Prasad.

**Specification.** `mate_choice_chinmay/RD_simulator_extensions_chinmay.docx` and `acceptance_tests_extensions.csv`.

**Commands.**
- `python scripts/run_chinmay.py` writes `outputs/chinmay_*.csv` and the figures.
- `python scripts/run_chinmay_power.py` writes `outputs/chinmay_power.csv`.

> **Read docs/07_leakage_audit.md with this file.** Most matches below are *by construction* or *input echoes*, and the CSV labels each row's evidence type. The one out-of-sample magnitude test is Khan's mating-bias score predicted blind from a Byrne & Rice 2006 calibration.

> **What the existing Manas core can say about this thesis: nothing.** The sexually antagonistic allele / Hill–Robertson / Robertson-covariance core has no infection state, courtship decision, fecundity cue or ejaculate effect. It emits none of courts-first (CF), courtship latency (CL), courts-most (CM) or a stepfather effect, and test A5 fails if it ever does. Every Chinmay statistic below is scored by the new modules only.

## 6.1 Source table (paper side; full version in `outputs/chinmay_sources.csv`)

"Thesis p.X" is the printed page number; the PDF page is given in brackets.

| Statistic | Location | Paper estimate |
|---|---|---|
| Courts first: 16 block × treatment binomial tests | thesis p.19 (PDF 37), Table 3.2 | all p > 0.05 (see flag C-F1) |
| Courts first: sham share, pooled | Table 3.2 (computed) | 240/463 = 0.518 |
| Latency: infection status | p.21 (PDF 39), Table 3.3 | F = 0.125, p = 0.723 |
| Latency: female genotype / male genotype | Table 3.3 | F = 6.93, p = 0.0087 / F = 6.49, p = 0.011 |
| Latency: interactions | Table 3.3 | p = 0.43–0.94 |
| Latency: means | Figs 3.2–3.3 | **MISSING** (figures only) |
| Courts most: 16 cells | p.21, Table 3.4 | means 0.432–0.637; SD 0.359–0.459; t-tests all p ≥ 0.067 |
| Courts most: ANOVA | p.22 (PDF 40), Table 3.5 | female genotype p = 0.249, male p = 0.916 |
| U mortality at OD600 1.5 | p.23 (PDF 41) | "around 70–90 %" |
| No fecundity cost in I/U/S | p.24 (PDF 42), citing Gupta 2016 | not remeasured on the choice females |
| Gupta 2016 fecundity, infected vs uninfected | Gupta PhD thesis p.133, Table 8.1(e) p.135 | treatment F = 1.83, p = 0.27; selection × treatment p = 0.025 (S infected > S uninfected); **measured 96 h post-infection on survivors** |
| Gupta 2016 survival, CFU | p.82, p.95 | I survival ~35–40% higher; I females ~30% fewer CFU |
| Telegony: gender (Wald) | p.31 (PDF 49), Table 4.2 | χ² = 108.2, p < 0.0001 (daughters better) |
| Telegony: stepfather population (Wald) | Table 4.2 | χ² = 1.02, df 2, p = 0.599 |
| Telegony: population × time × gender | Table 4.2 | p = 0.019 |
| Telegony: 6 h daughters | p.31, Table 4.1 | log-rank p = 0.047, Wilcoxon p = 0.020; the other 6 tests p ≥ 0.14 |
| Telegony: design | p.29 (PDF 47) | 50 infected + 30 sham per sex × stepfather × treatment; **one block** |
| Khan & Prasad 2013: mating-bias score | p.1021, Table 5 | 0.447 / 0.478 / 0.469 / 0.459 |
| Khan: fecundity of infected females | p.1019 §3.1 | "approximately 40% reduction" |
| Khan: copulation duration; mating latency | p.1019, Table 2 | no infection effect; 6.58 vs 6.22 min (p = 0.75) |
| Wittman & Fedorka 2015, experiment B (decapitated, infected vs unhandled) | J Insect Behav 28:41, Table 1B | CM 0.64 (P = 0.018); CF 0.58 (n.s.); latency n.s. |
| Morrow et al. 2003, *D. melanogaster* | Behav Ecol 14:803 | egg production F₅,₁₄₄ = 2.66, p = 0.025 (wounded laid fewest); leg-ablated females remated sooner |
| Crean et al. 2014 | Ecol Lett 17:1548, Table 1 | second male sired 87%; high-condition first male +0.5 SD offspring size (P = 0.013) |

**Flags**
- **C-F1.** The thesis's Table 3.2 p-values are the *smaller one-sided* binomial tail. This was verified for all 16 rows; for example 21/32 → 0.055 = P(X ≥ 21). The **two-sided** p-values are all ≥ 0.110 (closest: 0.110 and 0.185), so the null is stronger than the table suggests. The spec's A1 uses two-sided tests.
- **C-F2.** The thesis says "the sample size for each treatment was around 30" (p.17) and "50 vials" (p.16). Table 3.2 gives N = 17–39 courted vials per cell, and those N are used.
- **C-F3.** The fecundity evidence behind κ_IUS = 0 (Gupta 2016) was measured at 96 h on surviving females. Chinmay's choice females were tested at 12–14 h. Hence κ is implemented as a switch, graded "promising but limited".
- **C-F4.** The thesis's "p = 0.019" for the 6 h daughters (p.32) is the population × time × gender Wald term (Table 4.2). The log-rank p for that contrast is 0.047.

## 6.2 Parameter table (full version in `outputs/chinmay_parameters.csv`; config files `config/ius.yaml`, `config/khan.yaml`)

| Parameter | IUS | Khan | Provenance | Grade |
|---|---|---|---|---|
| Fecundity cost κ | 0 | 0.4 | IUS: Chinmay p.24 citing Gupta 2016 Table 8.1(e); Khan: "approximately 40% reduction" (Khan p.1019) | promising-limited / strongly supported |
| U mortality at OD600 1.5 | 0.80 | – | Chinmay p.23 "70–90 %" (midpoint; an input) | input |
| I vs U clearance | c_I = 0.66 (relative to U, derived) | – | Gupta 2016 p.82/95: I survival 35–40% higher → I mortality 0.425 | input |
| Courtship cost | implicit in depletion | – | Chinmay p.15–16 (12 h sperm depletion; Byrne & Rice 2006); spec: β = β₀ · depletion | ours |
| Depletion (depleted / not) | 1.0 | 0.5 | spec; direction from Byrne & Rice 2006, magnitude ours | ours |
| β₀ (kernel sensitivity) | 2.5 | 2.5 | frozen by the A2 pass (needs > 2.03); not biological | ours |
| Decapitation | on | off | Chinmay p.16 (Spieth 1966) / Khan §2.5; removes receptivity, not q | input |
| Cue perceptible | on | on | A4 turns it off | ours |
| Courts-most persistence | 0.95 | – | calibrated to the per-vial SD only (Table 3.4: 0.36–0.46) | ours |
| Compartmentalisation | on | – | Chinmay p.32: author hypothesis, flag only | speculative |
| Semen effect (flag off) | 0.5 SD | – | Crean et al. 2014 | strongly supported |
| Harm response ρ, λ | 0.5, 0.2 | – | signs from Morrow et al. 2003, magnitudes ours | established (sign) |
| Sex hazard ratio; daughter mortality | 2.0; 0.5 | – | sign from Chinmay Table 4.2; magnitudes **MISSING** (figures only) | ours |

**Not invented.** There is no fecundity curve, no dose–response curve and no latency means. The thesis's genotype effects on latency are left at 0 because their direction is not reported.

## 6.3 Comparison (from `outputs/chinmay_comparison.csv`; 200 replicate simulated studies, fixed seeds)

| id | Paper | Model | Verdict |
|---|---|---|---|
| A1-CF | sham first 0.518; all tests n.s. | sham first 0.50 [0.46, 0.55]. P(all 4 blocks n.s.) = 0.85; P(all 16 two-sided n.s.) = 0.67; P(all 16 one-sided n.s., the thesis metric) = 0.34 | **matches** (null) |
| A1-CL | infection p = 0.72 | median p 0.44; P(p > 0.2) = 0.71 | **matches** (null) |
| CL genotype | p = 0.0087, 0.011 | n.s. (intercepts 0) | **not identified** |
| A1-CM | means ~0.5 | 0.50 [0.46, 0.54] | **matches** (SD is calibrated) |
| A2 | sham first > 60% (spec) | 0.63 | **passes** |
| Khan bias | 0.447–0.478 | 0.41 [0.37, 0.46] | **matches sign, misses magnitude** (model stronger) |
| Khan copulation duration | no effect | difference −0.003; CI includes 0 in 95% of studies | **matches** (by construction) |
| A3 (prediction) | — | IUS with κ = 0.4. Spec-frozen β: sham first 0.73, detected in every study. **Byrne-calibrated β: sham first 0.56, detected by the pooled test in 73% of studies** | **prediction**, not a result; magnitude depends on β |
| Khan bias, blind (calibrated on Byrne & Rice only) | 0.463 | 0.480 [0.449, 0.502]. Raw +3.7%; effect vs 0.5: 0.020 vs 0.037 (−47%) | **the only out-of-sample magnitude test**; observed inside the interval |
| A4 (control) | — | κ = 0.4, cue off: 0.50 | no bias |
| Wittman & Fedorka | CM 0.64 | needs κ ≈ 0.5 with the same kernel | sign possible; magnitude **not identified** |
| B1 | harm alone not favoured | s = −0.17; mutant lost | **matches sign** |
| B2 | harm as a side effect | s = +0.25; fixes; female lifetime eggs 1.00 → 0.90 | **matches sign** |
| C1 sex | χ² 108, p < 0.0001 | detected in 100% of studies (χ² ≈ 43) | **matches sign**; magnitude not identified |
| C1 stepfather | p = 0.599 | median p 0.48; P(p < 0.05) = 0.04 | **matches** (null) |
| C2 | 6 h daughters p = 0.047 | **not predicted.** Under the null: P(this test p < 0.05) = 0.04; P(any of the 4 log-rank tests) = 0.20; P(any non-gender Wald term) = 0.24 | unreplicated; not a target |
| C3 (template) | Crean: P2 0.87, +0.5 SD | P2 0.87; first-male effect +0.49 (detected in 51% of 48-family studies); with the flag on, 0 | **matches** (inputs) |
| CORE | — | no CF/CL/CM/stepfather output | **not identified by the core** |

**Reading the A2 / Khan tension.** The spec's A2 threshold (sham courted first in more than 60% of trials) and Khan's measured group-assay bias (0.45–0.48) cannot both be met by one sensitivity β₀: β₀ ≈ 0.5–1.5 reproduces Khan but fails A2 (`outputs/chinmay_beta_tradeoff.csv`). Khan never scored courts-first. A spec revision could use "sham courted first significantly more than 50%" instead.

## 6.4 Diff against the Manas repository

**Added**

| Path | Contents |
|---|---|
| `extensions/__init__.py` | flags, `ModuleOff` |
| `extensions/config.py` | YAML-subset loader (pyyaml not installed) |
| `extensions/stats.py` | ANOVA, log-rank, Cox Wald; checked against scipy and against each other |
| `extensions/quality.py` | Module A: female quality |
| `extensions/choice.py` | Module A: courtship choice |
| `extensions/harm.py` | Module B |
| `extensions/telegony.py` | Module C |
| `config/ius.yaml`, `config/khan.yaml` | regime parameters with provenance |
| `scripts/run_chinmay.py`, `scripts/run_chinmay_power.py` | scoring and power runs |
| `tests/test_extensions.py` | 17 tests |
| `outputs/` | CSVs, figures, log |
| `docs/06_chinmay_extensions.md` | this file |

**Changed**
- `README.md`: a new section.
- `scripts/run_all.py`: a new `chinmay` stage.

No file under `iasc/` was modified. The core never imports `extensions/` (enforced by test A5).

**Tests**
- Before: `NUMBA_NUM_THREADS=2 python -m pytest -q tests/` → 8 passed.
- After: 25 passed (8 old + 17 new).
- B3 checks a SHA-256 of a fixed-seed engine run, and the assay $r_{w,g,mf}$ values, both captured before any extension code existed.

**What was not touched.** No Manas statistic was refitted. The rescaling rule and male recombination = 0 are unchanged. No deep learning, DGRP or DEST.

**On the Manas validations the spec lists as "must not change".** The Am Nat 2025 Robertson covariances were never accessible to this repo; it holds only a marked substitute (thesis Ch. 4). That entry therefore cannot change, because it was never scored.

## 6.5 What remains unpredicted, and the smallest assays that would decide it

**Not predicted or not identified**
1. **The latency genotype effects** (female p = 0.0087, male p = 0.011). Their direction is not given, and the model has no mechanism for them.
2. **Why there was no choice.** Two explanations give the same null, and Chinmay's data cannot separate them (A4 = A1):
   - the fecundity cue was absent (κ = 0);
   - males cannot perceive it.
3. **The size of Khan's bias.** The model's bias is stronger than Khan's under the A2-frozen β₀.
4. **The magnitude of the telegony sex effect**, and the 6 h daughter contrast. That contrast is deliberately not predicted; it occurs in ~20% of null replicates once all 4 log-rank tests are considered.

**Smallest extra assays**
- **Fecundity of the exact females used in the choice vials.** Measure eggs laid by infected and sham I/U females from the same cohort, starting 12–14 h post-infection: not 96 h, and not survivors only.
  - Chinmay's pooled courts-first data (240/463, exact 95% CI 0.472–0.565) bound β·κ ≤ 0.26.
  - With the male sensitivity calibrated independently on Byrne & Rice 2006 (docs/07), that excludes a perceived fecundity cost above about **44%** (30–86% over the 95% β range).
  - An earlier "~10%" figure used the β frozen by the spec's A2 threshold. That was circular and is withdrawn.
  - A Khan-sized 40% cost would have been detected in 73% of simulated studies, so Chinmay's null makes it unlikely but does not exclude it.
  - So: a measured cost near 0 supports "no fecundity difference". A measured cost of 40% or more points to males not perceiving it, or to weaker sensitivity than Byrne & Rice imply.
- **A replicated telegony block**, at least 3 more blocks with the same design.
  - One block detects a stepfather log-hazard shift of ±0.2 (about a 1.5× hazard ratio, I vs S stepchildren) with 74% power.
  - Four blocks detect ±0.1 (about 1.2×) with 80% power.
  - These figures use a baseline mortality we set ourselves; see `outputs/chinmay_power.csv`.
  - A replicated null would turn "promising but limited" into "strongly supported". A replicated 6 h daughter effect would need the compartmentalisation flag to be wrong for this trait.

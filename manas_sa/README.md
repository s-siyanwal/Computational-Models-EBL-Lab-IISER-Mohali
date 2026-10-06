# manas_sa: a hybrid model of Manas Geeta Arun's sexual-antagonism results

This repository simulates and analyses five results of **Manas Geeta Arun** (also M. A. Samant; ORCID 0000-0002-4433-4913), aiming to explain and predict them:

| # | Target | What is modelled |
|---|---|---|
| 1 | Geeta Arun *et al.* 2022, *BMC Ecol Evol* 22:38 | Intersexual genetic correlation for fitness (r_w,g,mf) at three adult sex ratios, from 39 LH hemigenomes |
| 2 | Samant 2022, PhD thesis (IISER Mohali) | Ch. 4: line variances, cross-sex trait–fitness covariances |
| 3 | Geeta Arun *et al.* 2025, *Am Nat* 206(6) (Robertson covariances) | **Not accessible.** Thesis Ch. 4 is used as a marked substitute, and no Am Nat number is quoted |
| 4 | Geeta Arun 2025, bioRxiv 2025.01.08.632022 ("Tie that binds") | Negative r_mf,W from purely sex-limited selection plus linkage (Hill–Robertson interference, HRI) |
| 5 | Singh, Jain, Geeta Arun & Prasad 2023, bioRxiv 2023.10.25.564090 | Imperfect modifiers resolving sexual conflict, autosome vs X |

The contrast set is Chippindale 2001, Innocenti & Morrow 2010, Collet 2016 and Ruzicka 2019 (LHM, negative or near-zero r). It is *reconciled*, not fitted.

Every statement is tagged as one of:
- **A**: reported in the paper;
- **B**: reproduced here;
- **C**: our extrapolation.

Verdict words are restricted to *matches sign / matches ordering / misses magnitude / not identified* (plus *null holds / fails*).

## Headline results (details in `docs/04_comparison.md`)

1. **Target 1 reproduced exactly from the raw data (B).** All 12 line-mean statistics match the paper to 4 decimals:
   - r_w,g,mf = 0.3805 / 0.4027 / 0.2515;
   - SA proportion;
   - all across-sex-ratio correlations;
   - the MB − FB difference, 0.129 (n.s.).

   This holds only with *untransformed* male proportions, although the Methods describe an arcsine transform (flag F12).
2. **The simulator reproduces the positive LH correlation (C).** It is an IBM with *Drosophila* transmission plus the emulated hemiclone assay, calibrated to variances only.
   - With mostly concordant variation (6% SA mutations, 93% of the rest concordant), a 39-line assay gives r = +0.45, against +0.40 in the paper.
   - It reproduces the sign and the sex-ratio ordering (FB lowest).
   - It predicts the male across-sex-ratio correlations out of sample (0.51 / 0.68 / 0.49 against 0.56 / 0.70 / 0.54).
   - Attenuation alone (better genetic signal at MB) predicts the observed MB − FB gap: +0.09 against +0.129.
   - It also predicts that the gap is non-significant with 39 lines (power 0.15); about 600 lines would give 80% power.
3. **The contrast set is reconciled by architecture and history, not by sampling or HRI (C).**
   - A few percent more SA mutational input (f_SA 0.10–0.19) moves the assay r to −0.24 … −0.35, the LHM range.
   - Near the boundary, replicate populations of one architecture differ by up to 0.9 in r.
   - Sampling SD at 40–223 lines (0.14–0.05) and linkage/HRI (population means −0.04 to −0.13 at 0.1–1 M; single populations down to −0.26) cannot bridge +0.4 to −0.5.
4. **Target 4 (HRI) sign and trend reproduced (B).** r_mf,W is negative and weakens monotonically with map length. At the preprint's own N (λ = 1) at 0.1 M: r = −0.042 ± 0.041, V_f = 0.018 (paper −0.078, V ≈ 0.02).
5. **Target 5 reproduced by re-running the authors' code (B).** 6 of 7 claims match (perfect modifier 97.2% vs 95.7%). The "autosomes beat the X at very high k₂" claim does not.
6. **The intuitive SA-vs-HRI test fails (C).** Recombining hemigenomes for up to 16 generations does not break HRI-type LD: power ≤ 0.18. The proposed discriminator is the N-dependence of r instead (docs/05).
7. **Methodological caution (C).** Without male recombination, SA or sex-limited variation can produce female-detrimental haplotype classes carried only by males. A pooled-sex genotype estimator of r_mf,W then reads intersexual differentiation as a negative genetic correlation: −0.94 vs a hemigenome r of −0.06 in such populations.
8. **Not identified:**
   - the Am Nat 2025 Robertson covariances (paper inaccessible);
   - trait-level covariances;
   - whether the sex-ratio effect is attenuation or a real G × SR change (39 lines cannot tell).

   The GP emulator failed validation (CV RMSE 0.30 vs replicate SD 0.15), so it is not used for inference.

## Extension: Chinmay (2019), mate choice and telegony (`extensions/`, off by default)

Temura Chinmay Krishna Yadav's BS–MS thesis (IISER Mohali 2019, supervisor N. G. Prasad) is **not** a sexually-antagonistic-allele study. The genetic core above cannot emit courts-first, courtship latency, courts-most or a stepfather effect (test A5). Three optional modules were added instead:

| Module | What it does | Default |
|---|---|---|
| quality | infection, clearance, fecundity cost κ, perceived quality, decapitation | off |
| choice | softmax courtship kernel, the Chinmay two-choice vial, the Khan group mating assay | off |
| harm | mate harm with the Morrow et al. 2003 female response; harm alone loses, harm plus mating success can spread | off |
| telegony | compartmentalisation switch: on for *Drosophila* (no stepfather effect), off is the *Telostylinus* template | off |

```bash
python scripts/run_chinmay.py          # outputs/chinmay_comparison.csv, _sources, _parameters, figures
python scripts/run_chinmay_power.py    # outputs/chinmay_power.csv
NUMBA_NUM_THREADS=2 python -m pytest -q tests/   # 25 tests (8 core + 17 extension, A1-A5, B1-B3, C1-C4)
```

**Results**
- **Chinmay's courtship null is a typical outcome when the fecundity cost is κ = 0.** In 85% of simulated studies all four blocks are n.s.
- **Khan & Prasad 2013 (positive control):** reproduced in sign only. The model's bias is stronger than Khan's under the β₀ that the spec's A2 threshold requires.
- **The telegony null is reproduced** with compartmentalisation on. The unreplicated 6 h daughter contrast is not predicted; a test like it comes out significant in ~20% of null replicates.
- **Only one result is a genuine out-of-sample magnitude test:** Khan's bias predicted blind from a Byrne & Rice calibration, 0.480 vs 0.463. The sign is right; the effect is about half the size, with the observation inside the predictive interval. The courtship and telegony nulls are by construction given κ = 0 and compartmentalisation.
- Details are in [`docs/06_chinmay_extensions.md`](docs/06_chinmay_extensions.md). The leakage, circularity and triviality audit for both projects is in [`docs/07_leakage_audit.md`](docs/07_leakage_audit.md).

**Smallest missing assays**
1. **Fecundity of the exact females used in the choice vials,** measured from 12–14 h post-infection. With the male sensitivity calibrated independently on Byrne & Rice 2006, the thesis's pooled courts-first data exclude a perceived cost above about **44%**. A Khan-sized 40% cost would have been detected in 73% of simulated studies, so it is unlikely but not excluded. An earlier "10%" figure was circular and is withdrawn (docs/07).
2. **A replicated telegony block.** The thesis ran one block. Four blocks detect a stepfather effect of about a 1.2× hazard ratio with 80% power.

## Biology in the model

**Genome and transmission**
- *D. melanogaster* genetics: XX/XY, **no crossing over in males**, female crossing over (Poisson), and X hemizygosity in males.
- Gene conversion is omitted and documented as such.
- Large N_e via documented rescaling (N/λ with U, R, s × λ; `docs/03`).

**Kinds of selection.** Each mutation has a pair of sex-specific effects:
- sex-limited;
- sexually concordant;
- sexually antagonistic (SA, intralocus conflict).

Interlocus conflict (sex ratio) acts as a per-sex intensity multiplier β_s(SR) on the measured fitness component. It is calibrated *only* to the line variances and within-cell noise in the raw data.

**Fitness components** are the assays' actual ones:
- eggs laid by 2 focal females in 18 h;
- proportion of progeny sired against LHst competitors at 1:3.

**Hemiclone transmission.** Each line is a male's sperm genome: his X plus a random autosome set. It is expressed on random backgrounds, hemizygous for the X in males.

**Statistics** are computed exactly as the papers do:
- line means and r_w,g,mf, the SA-variance proportion and across-sex-ratio correlations (target 1);
- the genotype-matrix decomposition COV = direct + indirect (LD) terms (target 4).

**Modifier layer.** A modifier on the autosome or the X (target 5) is run by re-executing the authors' own recursion grid.

## Layers

1. **Analytical** (`iasc/analytic/`):
   - exact two-sex, two-locus recursions with sex-specific recombination, autosomal or X;
   - the deterministic sex-differential LD result D* ≈ (1 − 2r̄)Δ_AΔ_B/(4r̄), which is negative for sex-limited loci and persists along a whole chromosome when r_m = 0 (C);
   - the authors' modifier model (target 5).
2. **Individual-based model** (`iasc/ibm/`):
   - a numba bitset Wright–Fisher simulator with sexes, X/Y, sex-specific recombination and finite sites;
   - the hemiclone-assay emulator, the papers' statistics, and the recombination experiment.

   SLiM 5.2 equivalents of the HRI model are in `slim/`. SLiM could not be built on this Windows machine (no MSYS2/WSL), so `scripts/slim_crosscheck.py` skips when `slim` is not on PATH.
3. **Emulator** (`iasc/emulator/`): a Gaussian process (plus a GBM comparison) trained on IBM design runs. It is used for sensitivity, inversion ("which SA fraction gives each published r") and power only.

## Run

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Python 3.14.6
python -m pytest -q tests/                      # 8 tests: theory checks, exact recomputation, zero leakage
python scripts/make_comparison.py              # rebuild results/comparison_table.csv from cached results (seconds)
python scripts/run_all.py --profile quick       # smoke run of every stage
python scripts/run_all.py --profile default     # laptop default, several hours on 6 threads, below-normal priority
```

- `--profile full` uses λ = 1 (the preprint's own N = 2,500) and 50 replicates. It is an overnight-plus run.
- Everything reads `config/default.toml`, and all seeds derive from `meta.master_seed`.
- Heavy jobs use (cpu − 2) threads at below-normal priority.

## Outputs, in the order requested

1. **Source map:** `docs/01_source_map.md`. Access status, extracted tables, flags F1–F12.
2. **Model spec:** `docs/02_model_spec.md`.
3. **Parameter table with provenance:** `docs/03_parameters.md`, including the λ decision.
4. **Comparison table:**
   - `results/comparison_table.csv` and `results/coverage.csv`;
   - narrative in `docs/04_comparison.md`.
5. **What the model cannot predict, plus the smallest experiment that distinguishes SA from HRI:** `docs/05_limits_and_distinguishing_experiment.md`.
6. **Code plus one command:** `python scripts/make_comparison.py`, or `scripts/run_all.py` to regenerate everything.

**Figures** (`results/`):

| File | Shows |
|---|---|
| `fig_r_vs_recombination.png` | r_mf,W vs map length, with and without male recombination |
| `fig_ld_vs_distance.png` | Signed LD between female- and male-limited alleles vs map distance |
| `fig_r_vs_sexratio_and_robertson.png` | r_w,g,mf vs sex ratio; within- vs cross-sex covariance of relative fitness |
| `fig_emulator_inversion_default.png` | Which SA fraction reproduces each published r |
| `fig_distinguish_default.png` | r after k generations of female meiosis: SA persists, LD decays |
| `fig_modifier_fraction.png` | Modifier resolution, autosome vs X |

## Layout

```
config/default.toml        single source of truth (values tagged [paper]/[data]/[bio]/[ours])
iasc/analytic/             recursions, modifier model
iasc/ibm/                  engine, architecture + rescaling, stats, hemiclone assay, experiments
iasc/estimators.py         the papers' estimators (applied to raw and simulated data alike)
iasc/emulator/             GP / GBM surrogate
analysis/                  recomputation of target 1 from the authors' raw data (10k bootstrap)
scripts/                   one script per stage + run_all.py + make_comparison.py
slim/                      SLiM 5.2 scripts (not executed here)
tests/                     theory, reproduction and zero-leakage tests
results/                   CSV tables, figures, logs; pops/ = saved evolved populations (.npz)
extensions/                optional Chinmay modules (quality, choice, harm, telegony); off by default
config/ius.yaml, khan.yaml regime parameters for the extensions, each with provenance
outputs/                   Chinmay comparison, sources, parameters, power, figures
```

**Zero leakage.** The IBM never imports the analytical layer (enforced by a test). Paper numbers appear only in `scripts/make_comparison.py`, `analysis/recompute_bmc2022.py`, `scripts/run_emulator.py` (published r values for inversion) and `docs/`. Calibration uses variances only.

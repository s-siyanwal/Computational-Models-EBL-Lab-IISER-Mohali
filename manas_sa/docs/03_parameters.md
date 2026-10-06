# 3. Parameter table with provenance

Every value lives in `config/default.toml`.

**Provenance key**
- **[paper]** = stated in the cited primary source.
- **[data]** = computed from deposited raw data.
- **[bio]** = textbook *D. melanogaster* biology.
- **[ours]** = a modelling choice, justified below.

## Reference HRI regime (target 4)

| Parameter | Value | Provenance |
|---|---|---|
| N (diploids) | 2,500 | [paper] Geeta Arun 2025, Simulations |
| Generations | 25,000 (= 10N) | [paper] |
| Autosome length | 1 Mb | [paper] |
| μ per bp per generation | 7×10⁻⁷ (sweep 5×10⁻⁷ … 1.3×10⁻⁶) | [paper] |
| Map length | 0.001, 0.01, 0.1, 1, 10 M | [paper] |
| DFE | reflected Gamma(shape 0.3, scale 0.05), all deleterious | [paper] |
| Dominance | 0.5 (additive) | [paper] |
| Fitness | w = exp(BV + N(0,1)); BV = Σ per-copy s of own-sex mutations | [paper] |
| Recombination by sex | both sexes (SLiM default) | [paper] implicit; flag F5 |
| Sample | 250 ♀ + 250 ♂ | [paper] |
| Replicates | 50 per condition | [paper]; default profile uses 6, full uses 50 |

## Drosophila-specific choices

| Parameter | Value | Provenance |
|---|---|---|
| Male crossing over | none (r_m = 0) | [bio] *D. melanogaster* males do not recombine (the clone-generator design relies on it: BMC 2022 Methods) |
| Gene conversion | not modelled | [ours] it adds only short-range exchange in females; it would weaken LD below about 1 kb, far below our map bins |
| X | hemizygous in males; female map 0.66 M | [bio] / [ours] map-length order from the Comeron *et al.* 2012 landscape (file `drosophila_genetics_models_papers_codes/file.pdf`) |
| Autosome map in the LH regime | 1.0 M (female) | [ours] about one chromosome arm |

## LH laboratory population (target 1 regime)

| Parameter | Value | Provenance |
|---|---|---|
| Adult census | 960 ♀ + 960 ♂, 60 vials × 16:16 | [paper] BMC 2022 Methods |
| Maintenance sex ratio | equal (the LH regime) | [paper] |
| Mutation-class mix ("shared") | SC 0.6, F-limited 0.15, M-limited 0.15, SA 0.10 | [ours]; varied by the emulator; no paper gives this |
| Mutation rate, DFE | as in the HRI reference | [ours] borrowed; no LH-specific estimate exists |
| SA benefit / cost ratio b | 0.5 | [ours]; b = 1 makes additive SA alleles exactly neutral, so they accumulate without bound (docs/02 §2.1) |
| Generations (reference scale) | 25,000 | [ours]; about 13N, long enough for mutation–selection–drift balance |
| Male map length | 0 | [bio] |

## Hemiclone assay (target 1 design)

| Parameter | Value | Provenance |
|---|---|---|
| Hemigenome lines | 39 | [paper] |
| Sex ratios (♂:♀ per vial) | 24:8, 16:16, 8:24 | [paper] |
| Focal:competitor | 1:3 | [paper] |
| Female assay | 7 vials × 2 days, eggs of 2 focal ♀ in 18 h | [paper] |
| Male assay | 5 vials × 2 days, 7 scored ♀ per vial, proportion sired | [paper] |
| Mean eggs per ♀; offspring per scored ♀ | 25; 20 | [ours] same order as the raw data |
| β_s(SR), noise sd | calibrated per sex × sex ratio | [data] targets only: line variance of relative fitness and within-cell CV² from Additional file 2 |

## Rescaling

The population is scaled by λ:

> N → N/λ; U, R, s → ×λ; generations → /λ

This preserves Nμ, Nr and Ns. The environmental noise sd stays at 1, which keeps the non-genetic N_e/N fixed (validated against the exact finite-N expectation).

**Known distortion (measured, not assumed).** Genetic variance in log-fitness scales as λ²: V_f ≈ 0.02 at λ = 1, 0.42 at λ = 4 and 3.4 at λ = 10. That adds variance in offspring number and strengthens background selection. The λ-sensitivity of r_mf,W is reported in the results: compare λ = 4 with λ = 2 and λ = 1 for 0.001 M and 0.1 M.

| Profile | λ | N (HRI) | Generations | Replicates |
|---|---|---|---|---|
| quick | 10 | 250 | 2,500 | 4 (smoke test only: degenerate at low recombination) |
| default | 2 | 1,250 | 12,500 | 6 |
| full | 1 (no rescaling) | 2,500 | 25,000 | 50 |

**λ decision (measured).**

| λ | Map length | Mean r_mf,W (reps) | Paper |
|---|---|---|---|
| 4 | 0.001 M | −0.934 (12) | median about −0.115 |
| 4 | 0.01 M | −0.9997 (12) | median about −0.125 |
| 2 | 0.01 M | −0.203 ± 0.081 (6) | median about −0.125 |
| 2 | 0.1 M | −0.152 ± 0.020 (6) | mean −0.078, median about −0.07 |
| 1 | 0.1 M | **−0.042 ± 0.041** (6); V_f = 0.018; replicates −0.22 to +0.06 | mean −0.078; V ≈ 0.02; range −0.30 to +0.125 |

At λ = 4 the populations split into two complementary haplotype classes:
- one female-lethal and male-good (frequency 0.25, carried only through sons, i.e. neo-Y-like);
- one complementary class (frequency 0.75).

The between-class share of load variance was 0.993 (`results/diag_haplotypes_map0.01_lam4.0.csv`). This is a rescaling artefact: Muller's-ratchet speed N·e^(−U/s) and N·V_g are not invariant. At λ = 2 there is no collapse: V_f ≈ 0.09 ≈ 4 × 0.02, as the λ² scaling predicts. The sign and the ordering across map lengths match the paper, but the magnitude is about 2× the preprint's. The λ = 1 check at 0.1 M measures that factor directly. At λ = 1 (the preprint's own parameters) r = −0.042 ± 0.041 and V_f = 0.018: the sign and V match, and the value lies inside the paper's replicate spread. At 0.1 M the λ = 2 runs give −0.086 to −0.152 depending on the seed set. HRI **magnitudes** from the λ = 2 grid are therefore read as upper bounds (about 2–3× too strong), while **sign and ordering** are trusted. Replicates that collapse at λ = 2 (4/6 at 0.001 M, 1/6 at 0.01 M) are counted and excluded; none of 18 collapse at λ = 1 (0.001, 0.01 and 0.1 M), which validates the filter. The unfiltered λ = 1 means are −0.184, −0.071 and −0.042 at 0.001, 0.01 and 0.1 M.

## Modifier layer (target 5)

All values are taken from the authors' code (supplement `media-1.docx`), **not** from the paper's Table 3. Flags F6–F9 explain the difference.

| Parameter | Value |
|---|---|
| s_f | {0.001, 0.2, 0.3, …, 1.0} |
| h | {0.1, 0.2, 0.3, 0.4, 0.49} |
| r_m, r_f | {0.1, 0.2, 0.3, 0.4} |
| s_m | Connallon–Clark window, step 0.1 |
| Initial B₂ | 0.05 |
| D | feasible range, step 0.01 |
| Generations | 3,000 |
| Fixation rule | frequency rounds to 1.000 |

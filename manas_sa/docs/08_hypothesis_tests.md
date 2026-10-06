# 8. Does the model predict or explain the results? Hypothesis tests

**Script:** `analysis/hypothesis_tests.py`, which writes `outputs/hypothesis_tests.csv` (300 replicate simulations per model, fixed seeds).

**Questions asked of every scorable result**
1. Is the observed value a plausible draw from the model's own sampling distribution? (predictive p-value)
2. Does it favour the model over a rival hypothesis or the trivial baseline? (likelihood ratio)
3. Is the test informative at all? A wide predictive interval passes almost anything.

**Multiple testing.** Benjamini–Hochberg FDR over the *substantive* tests only. Rows that cannot fail by design (by construction, input echoes, generic arithmetic) are reported but excluded, because they would dilute the family.

**Verdict levels**
- **REJECTED**: q < 0.05.
- **TENSION**: p < 0.05 but q ≥ 0.05.
- **consistent**: p ≥ 0.05.

Where the model's null distribution is known exactly (F for ANOVA terms, χ² for Wald terms), the analytic tail is used, so the Monte Carlo floor 1/(R+1) does not hide rejections.

## 8.1 Results

| id | Result | Test | p (q) | Verdict | Informative? |
|---|---|---|---|---|---|
| H1 | Chinmay courts-first 240/463 | likelihood ratio, κ = 0 vs κ = 0.4 (calibrated β) | LR = 3.4 | data favour **no fecundity cost**, moderately | yes, moderate |
| H2 | Chinmay courts-first share | predictive | 0.43 | consistent | **no** (by construction) |
| H3 | Courts-first heterogeneity across 16 cells | χ² dispersion | 0.70 (0.83) | consistent | yes (structure) |
| H4 | Latency: female genotype, F = 6.93 | F(1, ~900) tail | 0.009 (0.074) | **TENSION**: model has no genotype effects | yes |
| H4 | Latency: male genotype, F = 6.49 | same | 0.011 (0.074) | **TENSION** | yes |
| H4 | Latency: infection and interactions | same | 0.43–0.94 | consistent | no (by construction) |
| H5 | Courts-most cell means | standardised squared deviations | 0.90 | consistent | no (by construction + calibrated spread) |
| H6 | Courts-most SD, blocks 3–4 (held out) | predictive | 0.08 (0.21) | consistent; model 0.405 vs 0.423 | moderate |
| H7 | Khan Table 5 mean 0.463 | predictive under three models | calibrated 0.55; no-choice 0.14; spec β **0.028 (0.13)** | calibrated and no-choice consistent; spec-frozen β in **TENSION** (predicts too strong a bias, 0.413) | **weak** |
| H8 | Khan: model comparison | KDE log predictive density | — | LR calibrated / no-choice = **2.8**; calibrated / spec = **6.8** | weak / moderate |
| **H8b** | **Khan: between-vial SD implied by its CIs** | predictive, lower tail | **0.003 (0.045)** | **REJECTED**: data 0.029 vs model 0.111 [0.052, 0.177] | yes |
| H9 | Wittman & Fedorka, courted most 50/78 | binomial vs model P | κ = 0.4: 0.053; κ = 1: 0.30 | consistent only with an extreme cost (κ ≈ 1) | moderate |
| **H10** | **Telegony sex effect, χ² = 108** with the independent HR = 1 | χ²(1) tail | **2.5e-25** | **REJECTED** | yes |
| H10 | same, with HR = 2 (taken from the thesis) | predictive | 0.003 | rejected even with the echoed sign: model χ² ≤ 81 | n/a (input echo) |
| H11 | Telegony stepfather χ² = 1.02 | χ²(2) | 0.60 | consistent | no (by construction) |
| H12 | Smallest of 4 log-rank p = 0.047 | P under null | 0.23 | chance level | generic |
| H13 | **Manas: is the sex-ratio pattern attenuation only?** (data-only bootstrap) | disattenuated r: M 0.470, E 0.481, F 0.349 | MB−FB 0.34; E−FB 0.30 | not rejected | **low power** (the raw difference is n.s.) |
| H14 | Manas $r_{w,g,mf}$ (M/E/F) under the shared regime | predictive | 0.17 / 0.08 / 0.34 | consistent | **no**: 95% interval spans about −0.31 to +0.48 |
| H14 | Male across-SR r | predictive | 0.73–0.90 | consistent | weak (calibrated noise) |
| H14 | Female across-SR r (M–F, M–E) | predictive | 0.050 (0.18) / **0.017 (0.090)** | **TENSION**: model too high (no G×SR) | yes |
| H15 | Manas r(E) = 0.40 over the architecture design | prior predictive | 0.16 | consistent | **no** (design spans −0.8 to +0.5) |
| H16 | HRI mean −0.078 at 0.1 M (λ = 1) | Welch t (paper SD approximated) | 0.44 | consistent | **weak**: n = 6; H17 shows the model mean itself is not distinguishable from 0 |
| **H18** | **HRI collapse at λ = 1, 0.001 / 0.01 M** | count of replicates with V_f/λ² > 0.1 | — | **0/6 and 0/6 collapsed** (V_f 0.014–0.037): the λ = 2 collapse is a rescaling artefact, so the filter is justified | yes |
| H19 | HRI median −0.115 (0.001 M), −0.125 (0.01 M) at λ = 1, **no filter** | Welch t (paper SD approximated) | 0.25 / 0.42 | consistent: model −0.184 ± 0.052 and −0.071 ± 0.061 | moderate (n = 6 each) |

## 8.2 What the tests say

**1. Where the model genuinely predicts, the evidence is weak but in the right direction.**
- **Khan (H7/H8).** The independently calibrated model is preferred over the no-choice baseline, but only by LR ≈ 2.8. It is preferred over the spec-frozen sensitivity by 6.8.
- **Chinmay (H1).** His courts-first count favours "no fecundity cost" over a Khan-sized cost by LR ≈ 3.4.
- Neither study has the power to establish more.

**2. The model is rejected, or under tension, wherever the data contain structure it lacks.**
- **Khan's between-vial dispersion (H8b, rejected).** The model with ≤ 10 single matings per vial predicts much more vial-to-vial variation than Khan's confidence intervals imply. Either the CIs are not vial-based, or more matings per vial occurred than the protocol suggests. Either way, the model's assay mechanics do not explain it.
- **Telegony sex effect (H10, rejected).** It cannot be produced from independent data (Gupta: no sex difference). Even when the sign is taken from the thesis, the model's χ² never reaches 108.
- **Latency genotype effects (H4, tension).** The model has none.
- **Female across-sex-ratio correlations (H14, tension).** They are lower than the model's. The model's only sex-ratio channel is attenuation, which cannot lower a correlation that much, so this points to a real genotype × sex-ratio interaction in female fitness.

**3. Many "consistent" verdicts carry no information.**
- Courtship nulls, the stepfather null, and courts-most means are consistent by construction.
- The Manas $r_{w,g,mf}$ values are consistent only because the shared regime's predictive interval is about 0.8 wide.
- The architecture design spans the whole published range (H15), so any value is "consistent".
- The HRI magnitude check (H16) has n = 6 and cannot distinguish the model from zero (H17).

**5. The post-hoc HRI collapse filter is validated (H18, H19).** At the preprint's own scale (λ = 1), none of 12 short-map replicates collapses, so the λ = 2 collapses (4/6 at 0.001 M, 1/6 at 0.01 M) are artefacts of rescaling, not biology. The unfiltered λ = 1 means (−0.184, −0.071) agree with the paper's medians (−0.115, −0.125) within sampling error. Across map lengths the λ = 1 means are −0.184, −0.071, −0.042 at 0.001, 0.01, 0.1 M: the ordering of the paper. Each mean rests on 6 replicates, so magnitudes are only roughly known (SE ≈ 0.05–0.06).

**4. The attenuation explanation of the sex-ratio effect (H13).** Corrected for reliability, r is 0.47 / 0.48 / 0.35 (M / E / F). Equality is not rejected (p ≈ 0.3), but the test has little power, because the raw MB−FB difference was itself non-significant. Attenuation is *sufficient* for the published pattern; it is not *demonstrated*.

## 8.3 Summary scorecard

| Category | Count |
|---|---|
| Genuine out-of-sample predictions consistent with data | Khan sign (weak evidence); held-out CM spread; HRI sign, ordering and short-map magnitude at λ = 1 (replication, no filter) |
| Rejected or tension (the model lacks a mechanism, or it was mis-specified) | Khan vial dispersion; telegony sex magnitude; latency genotype effects; female across-SR correlations; Khan under the spec-frozen β |
| Consistent but uninformative or by construction | most of the remaining rows |
| Audit validations | HRI collapse filter justified (0/12 collapse at λ = 1) |
| Hypothesis comparisons | no fecundity cost over Khan-sized cost (LR 3.4); calibrated sensitivity over spec-frozen (6.8); over no choice (2.8) |

## 8.4 Coupled model (docs/09)

The tests above score the assay layer with global constants. The coupled model (`scripts/run_coupled.py` →
`outputs/coupled_comparison.csv`) re-scores Chinmay A1 and Khan A2 with genotype-specific κ and β taken from an
evolved IUS population. Its verdicts carry their own `evidence_type`. Hand-set κ = 0 stays **by construction**,
and a prior-dependent scenario is never relabelled as an out-of-sample genetic prediction. The H8b, H10, H4 and H14
verdicts above are unchanged.

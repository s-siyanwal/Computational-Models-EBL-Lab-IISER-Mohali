# 7. Leakage, circularity and triviality audit

**Question.** Could any "match" between model and paper have been produced by giving the model the answer? That covers six routes:
1. target numbers used as inputs;
2. calibration on the same data that is scored;
3. choices made after seeing the comparison;
4. structural assumptions that guarantee the outcome;
5. selection or exclusion rules defined post hoc;
6. criteria a no-model baseline would also pass.

**Evidence types.** Every row of `outputs/chinmay_comparison.csv` now carries an `evidence_type` and a `leakage_note`. The categories, from weakest to strongest:

| Type | Meaning |
|---|---|
| by construction | the assumptions guarantee the result; nothing is tested |
| input echo | a value taken from the target (or the same source) is compared with itself |
| fitted | a free parameter was tuned to the target, or to a threshold derived from it |
| calibrated | partly uses target data (e.g. variance-only calibration); the match is partly inherited |
| existential | true for some parameter values that were chosen to make it true |
| out-of-sample prediction | no information from the target entered the model |
| prediction (unmeasured) | concerns a condition nobody measured; cannot be scored |

## 7.1 Chinmay extension

| Claim | Evidence type | Why |
|---|---|---|
| Courts-first, latency and courts-most nulls (A1) | **by construction** | κ = 0 makes the kernel exactly 0.5; latency depends only on max(q), which is equal for both females. Only binomial and ANOVA sampling is simulated, and that holds for any null. The scientific content is in the **input** κ = 0, which comes from Gupta 2016. That source is independent of Chinmay, but it measured fecundity at 96 h on survivors, not on the 12–14 h choice females. |
| Courts-most spread | **calibrated on the target** | The persistence parameter was fitted to the per-vial SD in the same Table 3.4 that holds the scored t-tests. |
| A2 threshold passed | **fitted** | β₀ was chosen to clear the spec's 60%, so passing was guaranteed. |
| Khan bias with the spec β₀ | **contaminated** | β₀ was moved from 3.0 to 2.5 *after* seeing the Khan comparison (bias 0.395 → 0.411). Disclosed; superseded by the blind version below. |
| **Khan bias, blind** | **out-of-sample** | Kernel calibrated only on Byrne & Rice 2006 (fecundity measured: Δq = 0.638; vial sign-test counts 22/32 and 57/72). Khan supplies only κ (its own fecundity measurement) and the assay design. A test asserts that the fitting code never reads Khan's Table 5. |
| Copulation duration, cue-off control (A4) | **by construction** | Independent draw / equal q. |
| Wittman & Fedorka | **fitted (inverse)** | κ was solved to reproduce CM 0.64. With the calibrated β, no κ ≤ 1 reaches it. |
| Harm-only mutant loses (B1) | **by construction** | With ρ, λ ≥ 0 and P2 < 1, harm can only lower the eggs a male sires per mating. This restates Morrow's empirical signs; it is not an independent result. |
| Harm-plus-competitiveness spreads (B2) | **existential** | True for any m > m*(η); m = 0.5 was chosen. The real content is the boundary m*(0.5) = 0.20, and its magnitude rests on our ρ and λ. |
| Telegony sex effect (C1) | **input echo, circular** | The sign of the sex hazard ratio was set *from this very result*. The only independent data (Gupta 2016 p.82: about 60% survival in both sexes at 12 d, IUS flies) do not support a sex difference. |
| No stepfather effect (C1) | **by construction** | The stepfather effect is exactly 0 when compartmentalised. |
| 6 h stepdaughter contrast (C2) | **generic** | Multiple-testing arithmetic that holds under any null. |
| Telostylinus template (C3) | **input echo** | Crean's P2 = 0.87 and +0.5 SD go in, and come out. |
| Chinmay's null bounds the fecundity cost below ~10% | **circular, now withdrawn** | It used the β frozen by the spec's A2 threshold. With the independently calibrated β the bound is **κ ≤ 0.44** (0.30–0.86 over the 95% β range). |
| A3 prediction | **prediction**, β-dependent | Spec β: a 40% cost would always be detected. Calibrated β: detected by the pooled 463-vial test in **73%** of simulated studies. |

**Net.** The Chinmay extension has **one** out-of-sample magnitude test (Khan's blind bias) and one counterfactual prediction (A3, calibrated). Everything else is a consistency check, an echo of an input, or a consequence of the assumptions. The courtship null is not explained by the model: it is *assumed* through κ = 0, and the model shows that this assumption is sufficient. A4 shows the null is equally consistent with males failing to perceive the cue.

## 7.2 Can the magnitude be brought within 10%?

| Item | Legitimate route | Result | Within 10%? |
|---|---|---|---|
| Khan mating-bias score | independent calibration on Byrne & Rice 2006 | 0.480 [0.449, 0.502] vs 0.463 | **Raw scale +3.7%, yes, but the no-choice baseline 0.500 is also within 10% (+8%).** On the effect scale (deviation from 0.5): 0.020 vs 0.037, **−47%**, though the observed value lies inside the prediction interval. Log-quality scale: 0.484. |
| Khan, by fitting β to Khan | fitting to the target | β₀ ≈ 1 gives 0.462 | yes, **but circular**; it would also make A2 fail. Not adopted. |
| Khan mating latency | independent Byrne & Rice controls support a quality slope ≈ 0 | ratio 1.00 vs 1.06 / 1.02 | yes, but this is the trivial no-effect baseline |
| Wittman & Fedorka CM 0.64 | calibrated β | maximum 0.575 at κ = 1 | **no**, and not reachable by any fecundity cost |
| Telegony sex χ² | no independent hazard ratio exists; Gupta's data suggest ≈ 1 | — | **not identified**; any match would be a fit to the target |
| Byrne-calibrated A3, κ bound | — | predictions; no target exists | n/a |

**Conclusion.** A 10% criterion on raw proportions near 0.5 is uninformative, because the no-effect baseline passes it. On effect sizes, the one genuine out-of-sample test gets the sign right and the magnitude **about half the observed**, with the observation inside the predictive interval. No other sign-matched quantity can be brought within 10% without fitting to the target it is scored against.

## 7.3 Manas core (earlier work)

| Claim | Evidence type | Diagnostic |
|---|---|---|
| Table 2A recomputed exactly | reproduction, estimator variant selected to match | The no-arcsine variant was chosen *because* it reproduces 12/12 published values (flag F12). That identifies what the authors did; it is not a model success. |
| SA proportion "reproduced" | **algebraic identity** | It equals (1 − r)/2 for standardised line means, so it carries no information beyond r. |
| Sex-ratio ordering of $r_{w,g,mf}$ (FB lowest) | **echo of the calibration targets** | The sex-ratio effect enters only through β_s(SR) and noise, which are calibrated to line variances and within-cell CV² from the same dataset. **Data-only check, no simulator:** reliability from the raw variance components gives attenuation factors E 0.837 > M 0.809 > F 0.721, already the published ordering. Quantitatively, attenuation explains r(FB) ≈ 0.35 of the observed 0.25 (about a third of the drop); the rest is not explained. |
| LH value +0.40 reproduced (design point 8, +0.45) | **fitted by selection** | One of 12 design points was picked *because* it matched. The architecture is fitted. The "LH-like" power analysis conditions on that fit. |
| Male across-sex-ratio correlations (0.51 / 0.68 / 0.49 vs 0.56 / 0.70 / 0.54) | partly calibrated | Noise levels come from within-cell CV² of the same data, so reliability is inherited. The match of *level* is therefore weak evidence; the female mismatch (0.87–0.89 vs 0.75–0.84) is the informative part. |
| Contrast reconciliation (SA share flips the sign) | exploratory sweep | Describes the model's response surface; no target was fitted. The LH/LHM values fall inside the range explored, which a broad sweep makes easy. |
| HRI sign and trend | **replication of the same model** | Fitness function, DFE and parameters were copied from the preprint, so agreement is expected if the code is right. It checks the implementation, not the biology. |
| HRI magnitudes at λ = 2 | **post-hoc exclusion rule** | The collapse filter (V_f/λ² > 0.1) was defined after seeing collapsed replicates. It moved 0.001 M from −0.75 (all replicates) to −0.26. It is justified only by λ = 1 having no collapse, and λ = 1 was checked **at 0.1 M only**. The 0.001 M and 0.01 M magnitudes are therefore unvalidated. **Resolved in §7.5:** λ = 1 runs at both map lengths show 0/12 collapses. |
| λ choice | partly outcome-driven | λ = 4 was rejected after it gave r ≈ −1 (a degenerate state, but also a mismatch). The λ = 1 check at 0.1 M is clean. |
| SA benefit ratio 0.5; f_SA ≤ 0.2 | post-hoc model restrictions | Introduced after runaway and collapse. They were not tuned to a target value, but they shape every SA-regime result. |
| Modifier claims (6 of 7) | **re-execution of the authors' code** | Reproduces their computation by construction. Discrepancies (+1.5%, the one failed claim) are the informative part. |
| Distinguishing-experiment conclusions | in-silico, assumption-driven | Results follow from the IBM's LD structure. Power numbers use a baseline mortality we set ourselves. |

## 7.4 Rules adopted going forward

1. **Report effect sizes against a null baseline,** never raw proportions near 0.5.
2. **Label the evidence type of every comparison row** (now in the CSV).
3. **Calibrate sensitivities on independent studies** with measured inputs (Byrne & Rice), never on the scored target or on thresholds derived from it.
4. **Freeze exclusion rules before looking,** or report both filtered and unfiltered summaries. The HRI table reports collapse counts; the unfiltered means are in `results/hri_runs_default.csv`.
5. **Keep target data out of fitting code by structure,** and test for it (`test_independent_calibration_never_reads_khan_or_chinmay_targets`).

## 7.5 Fixes applied (second pass)

| Finding | Fix | Effect |
|---|---|---|
| β₀ fitted to the spec's A2 threshold, then moved after seeing Khan | Default β now calibrated **only** on Byrne & Rice 2006 (depleted 0.59, non-depleted 0.32; `config/ius.yaml`). The spec value lives in `config/spec_a2.yaml`, used only by the A2 compliance test. A test checks the config against the calibration record. | Khan becomes an out-of-sample test (sign right, effect ~2/3). A2 holds only with the labelled spec override. |
| Courts-most spread calibrated on the scored table | Persistence calibrated on Table 3.4 blocks 1–2; blocks 3–4 held out | Held-out SD 0.405 vs observed 0.423 (−4%, predictive p = 0.08) |
| Latency–quality slope assumed | Set to 0, as supported by Byrne & Rice no-choice controls (spec value kept in the override) | The Khan latency "match" is now explicitly the null baseline |
| Verdicts calling by-construction results "matches" | Verdict text overridden: "consistent (by construction, not evidence)"; input echoes "not scored" | — |
| Telegony sex effect echoed from the target | Not scored. The test with the independent HR = 1 is reported as **rejected** (H10) | — |
| "Fecundity cost < ~10%" bound | Withdrawn. Calibrated bound: κ ≤ 0.44 | A3: a 40% cost would be detected in 73% of studies, not 100% |
| B2 existential | Invasion boundary m*(η) reported and tested | — |
| SA proportion as evidence | Labelled as the identity (1 − r)/2 in `results/comparison_table.csv` | — |
| Sex-ratio ordering as a prediction | Labelled an echo of the calibration; data-only attenuation test added (H13) | — |
| LH +0.45 as a prediction | Labelled "selected from the sweep = fitted"; prior-predictive test H15 shows it is uninformative | — |
| Post-hoc HRI collapse filter | λ = 1 runs at 0.001 and 0.01 M to check whether collapse occurs at full scale (H18; `results/lambda_sensitivity.csv`) | **Filter justified:** 0/6 collapsed at each map length (V_f 0.014–0.037). Unfiltered λ = 1 means −0.184 ± 0.052 (0.001 M) and −0.071 ± 0.061 (0.01 M) vs paper medians −0.115 and −0.125 (H19: p = 0.25, 0.42). The λ = 1 values are now the headline; λ = 2 values remain upper bounds. |
| FDR diluted by trivial rows | FDR family restricted to substantive tests | Two rejections and three tensions surface (docs/08) |

**Not fixable with the available data.**
- Latency genotype effects: direction not reported.
- Khan between-vial dispersion: the CI construction is unknown.
- Telegony sex magnitude: no independent hazard ratio exists.
- The LH mutation architecture: no measurement exists.

These remain **not identified** or **rejected**, and are reported as such.

## 7.6 Coupled model (docs/09)

The coupled model (genetic layer → genotype record → assay layer) leaves every label above in place.
- **Still by construction:** the courtship nulls when κ = 0 is set by hand (scenario `kappa0`).
- **Stay rejected or in tension:** H8b, H10, H4 and H14.
- **Male recombination:** audited. All Drosophila runs have it off; the True rows replicate the preprint's own
  SLiM setting.

Row-by-row evidence types are in `outputs/coupled_comparison.csv`.

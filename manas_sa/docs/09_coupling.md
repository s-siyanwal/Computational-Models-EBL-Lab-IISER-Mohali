# 9. Coupled model: genetic layer → genotype record → assay layer

The two layers stay separate programs. The **genetic layer** (`iasc/`) evolves populations. Each generation it writes
a **genotype record**. The **assay layer** (`extensions/`) reads a record through the **coupler**
(`extensions/coupler.py`) and runs the audited assays (`quality`, `choice`, `harm`, `telegony`).

- **One direction.** Nothing in `extensions/` writes into a record or a population. Nothing in `iasc/` imports
  `extensions/` (test A5).
- **Exception.** An evolutionary-choice run, where mate choice feeds back into reproduction, is the only
  allowed exception. None is implemented, so no such run exists.
- **No refitting.** Neither layer is refitted to the other:
  - β keeps its Byrne & Rice calibration, and the record supplies only each genotype's value relative to it;
  - no genetic-layer parameter is tuned to an assay outcome;
  - the Manas r_mf,W results are untouched.
- **Flags.** Every coupling flag defaults to off (`coupling: false`). With it off, the assay layer reads its global
  constants exactly as before (regression test 1).

## 9.1 The genotype record

Written by `iasc/ibm/regime.py` (genetic layer). There are two outputs:
- a per-generation summary (`results/coupled/<run>_generations.csv`);
- a per-individual record of the evolved populations (`results/coupled/<run>_record.csv`).

| Field | Meaning | Who computes it | Source of the numbers |
|---|---|---|---|
| `population`, `block`, `regime`, `generation`, `sex`, `id` | provenance of the genotype | genetic layer | run settings |
| `clear_rel` = c(G)/c̄_anc | clearance after *P. entomophila*, relative to the ancestral mean (I- vs U-derived via `regime`) | genetic layer: exp(additive BV over clearance sites) | mutational effects: **prior [ours]**. Absolute decay rate of the load-driven cost per hour (`cost_decay`) is **MISSING** (scenario parameter) |
| `tol_rel` = a(G)/a₀ | fecundity sensitivity to pathogen load (tolerance), relative to the ancestral mean | genetic layer | prior [ours]; a₀ is a scenario parameter |
| `kappa_13h`, `kappa_lay` | κ(G, dose, hours) = 1 − exp(−a₀·tol_rel·D^ν·exp(−cost_decay·clear_rel·t)), with a₀ set so the ancestor's 96 h cost equals the scenario value at the assay time (13 h) and the egg-laying time (105 h) | genetic layer (formula of the regime) | ν = 2.08, derived from two inputs: 50% mortality at OD 1.0 (Gupta 2016) and 80% at OD 1.5 (Chinmay p.23). a₀ and cost_decay are scenario parameters, **MISSING** |
| `H_f`, `H_m` | sex-specific cumulative 96 h hazard at the assay dose: H₁·D^ν / c(G) × sex factor | genetic layer | H₁ = −ln 0.5 at OD 1.0 (input). **Sex factor = 1**: no independent data give a genetic sex difference. Gupta 2016 does not supply one, and Chinmay Table 4.2 is the scored target (H10) |
| `beta_rel` | male preference slope relative to the calibration population: π(G) = max(0, 1 + BV_b − mean_anc BV_b). **0 = cannot perceive infection** | genetic layer | prior [ours]. The assay layer turns it into β(G) = β_Byrne-Rice × depletion × π(G) |
| `w_f`, `w_m` | sex-specific adult fitness from the existing Manas function, exp(BV_f) and exp(BV_m) (no noise) | genetic layer (unchanged fitness function) | Manas architecture (`config/default.toml` [lh]) |
| `sa_load_f`, `sa_load_m` | contribution of the SA sites (kinds 4/5) to BV_f and BV_m | genetic layer | as above |

κ = 0 and β = 0 are legal genotypes (a(G) = 0, or π(G) = 0). They are no longer global constants in a coupled run.

**Drosophila constraints in the genetic layer**
- male recombination is 0 (asserted by the runner, regression test 4);
- the X is hemizygous in males;
- the ancestral burn-in is rescaled by λ = 2 from the BRB census (N ≈ 2500, Gupta 2016), preserving N·μ, N·r and
  N·s;
- the selection phase runs at the IUS census (150 + 150 adults infected per generation, Chinmay p.7, PDF p.25), so λ = 1.

## 9.2 IUS regime in the genetic layer

The protocol inputs come from Chinmay 2019 p.7 (PDF p.25) and Gupta 2016 Ch. 6.

**I regime, each generation**
- All adults are infected on day 12.
- Survivors of the 96 h window breed. Survival ~ Bernoulli(exp(−H_sex(G, D_t))).
- The dose D_t is reset every generation so that mean survival is 0.5, as in the protocol (Gupta: "to keep
  percentage mortality at the desired level").
- A female's offspring share is multiplied by 1 − κ(G, D_t, 105 h): she lays in the 18 h window after 96 h.
- Adult fitness is the Manas exp(BV + N(0, 1)) on top of survival.

**U regime.** Unhandled: no infection, so clearance, tolerance and preference sites are neutral. With zero trait
effects the U generation is bit-identical to `iasc.ibm.engine.evolve_one`, the same code path as every Manas
result (regression test).

**Preference sites.** These are neutral in both regimes. In the I regime every breeding female is a survivor of the
same infection, so a male preference for uninfected females has no target.

**Generations.** T = 60 generations of selection. This is a lower bound: Gupta 2016 reports "more than 60
generations" by January 2016. The generation at Chinmay's 2019 assays is **MISSING**.

## 9.3 What the coupling may claim

| Row | Rule | Consequence in `outputs/coupled_comparison.csv` |
|---|---|---|
| **Chinmay A1** (CF, CL infection term, CM nulls) | Explained only if I and U genotypes evolved under IUS give κ(13 h) ≈ 0 or β ≈ 0 **and** the assay then matches. | U is unselected for clearance, tolerance and preference, so U genotypes keep the ancestral κ(13 h) and β. A null in U×U vials therefore needs an *ancestral* κ(13 h) ≈ 0 or β ≈ 0, which no IUS selection produces. Scenario a₀ = 0 is labelled **by construction**. Scenarios a₀ > 0 are labelled **prediction (prior-dependent)** and scored against the thesis. |
| **Khan A2** | Different regime (Serratia, κ measured 0.4, intact females). β from Byrne & Rice. Never retuned on Khan Table 5. | β varies by genotype around the calibrated mean (π(G) from the ancestor record): **out-of-sample**. |
| **SA alleles and courtship** | An SA allele may change courtship only through κ, β or the fecundity cue. | The cue is q = 1 − κ by default, so baseline fecundity differences are not perceived. Option `cue_includes_baseline_fecundity` (off) is the documented third channel. Regression test: changing SA genotypes with κ and β fixed leaves CF, CL and CM bit-identical. |
| **Telegony** | Non-genetic. With compartmentalisation on, the stepfather's genotype (SA included) cannot shift offspring survival. | The coupler passes nothing to `telegony.py`. Test: the telegony output is identical whatever stepfather record is supplied. Mother's curse is not used. |

## 9.4 Audit rows: which change evidence type

| Audit row (docs/07, docs/08) | Before coupling | With coupling |
|---|---|---|
| Courts-first, latency infection term, courts-most nulls (A1) | by construction (κ = 0) | **still by construction** when a₀ = 0. With a₀ > 0: prediction (prior-dependent), not explained by IUS evolution (see 9.3) |
| Female genotype × infection interaction in CF and CL | not modelled | **new derived prediction**: if a₀ > 0, I females clear faster, so κ_I(13 h) < κ_U(13 h) and the CF bias is weaker in I-female vials. Sign derived; magnitude prior-dependent |
| Male genotype × infection interaction | not modelled | **derived null**: preference is neutral in both regimes, so I and U males differ only by drift |
| Khan bias (H7/H8) | out-of-sample, β fixed | out-of-sample, β with genetic variance (same calibrated mean) |
| **H8b** Khan between-vial SD 0.029 vs 0.111 | REJECTED | **stays REJECTED.** Genetic variance in β *adds* vial-to-vial variance, so it cannot be hidden in the genetic layer. A fix must come from the mating scheme or the CI interpretation |
| **H10** telegony sex χ² = 108 | REJECTED | **stays REJECTED.** Record sex factor = 1 (no independent data). Gupta 2016 does not supply a genetic sex difference. If taken from Chinmay Table 4, it is an input echo and is not scored |
| **H4** latency genotype main effects | TENSION | **stays.** A genetic intercept per sex genotype is allowed, but no latency trait is modelled, so it is not identified. No infection interaction is added by hand |
| **H14** female across-sex-ratio correlations | TENSION | **stays.** No genotype × sex-ratio term is added. If one is added later, it is reported as a new prediction, not a fix of the old interval |
| Survival advantage of I (0.375), I-female CFU 30% lower (Gupta) | input to the quality module | **existential / prior-dependent**: the evolved values are reported but depend on the unmeasured mutational variance and T. Not fitted |
| HRI male recombination | — | **Audited.** Every Drosophila run (`hri_drosophila_*`, LH, design) has male_recombination = False. The True rows (`maplen/paper`, `mu_sweep/paper`, `lambda*`, `ne_check`) replicate the preprint's own SLiM model, which used the default Wright–Fisher mode in which both sexes recombine (preprint Methods). No Drosophila run had the wrong flag, so no rerun was needed and H19 is unchanged. Regression test 4 enforces this |

## 9.5 Results

**Run:** `python scripts/run_coupled.py`, which writes `outputs/coupled_comparison.csv` (200 replicate studies per
scenario).
- **Ancestor:** burn-in at N = 1250 and λ = 2 for 5000 generations (134 s).
- **Selection:** 4 blocks × (I, U) × 60 generations at 150 + 150 per scenario.
- **Sensitivity run:** `--no-sd-scaling`, written to `outputs/coupled_comparison_sens_sd_realised.csv`. It uses
  the raw, lower standing variance (see 9.6).

**Genetic layer (evolved I vs U, main run)**

| Scenario (ancestral κ at 96 h; cost decay) | κ(13 h) I / U / ancestor | β_rel I / U | survival advantage at OD 1.5 (Gupta: 0.35–0.40) | load I/U (Gupta: ≈ 0.7) |
|---|---|---|---|---|
| `kappa0` (κ = 0 set by hand) | 0 / 0 / 0 | 1.06 / 1.06 | +0.22 | 0.49 |
| `k96_0.1_flat` | 0.096 / 0.105 / 0.100 | 1.06 / 1.08 | +0.28 | 0.41 |
| `k96_0.1_decay24h` | 0.82 / 0.97 / 0.97 | 0.99 / 1.09 | +0.28 | 0.43 |
| `k96_0.3_flat` | 0.28 / 0.31 / 0.30 | 0.95 / 0.98 | +0.27 | 0.43 |

- **Survival and load.** I evolves higher clearance in every scenario, so the sign of the survival advantage and
  the direction of the load difference are derived.
- **Magnitudes are prior-dependent.** With the raw standing variance (sensitivity run) the advantage is only
  +0.01 to +0.02 and the load ratio 0.92–0.96. The main run's +0.22 to +0.28 and 0.41–0.49 therefore reflect the
  chosen ancestral SD of 0.15, not data. Neither was fitted.
- **U keeps the ancestral κ(13 h)** in every scenario, and β stays near 1 in both regimes (drift only).

**Assay layer on the evolved genotypes**

| Scenario | CF share (obs 0.518) | P(all 4 blocks n.s.) | latency INF F (obs 0.125) | A1 explained by IUS evolution? | evidence type |
|---|---|---|---|---|---|
| `kappa0` | 0.501 [0.460, 0.540], p = 0.49 | 0.84 | median 0.57, p = 0.48 | **no** (κ = 0 hand-set) | by construction |
| `k96_0.1_flat` | 0.518 [0.467, 0.562], p = 1.0 | 0.76 | 0.74, p = 0.44 | **no** | prediction, prior-dependent |
| `k96_0.3_flat` | 0.542 [0.503, 0.588], p = 0.37 | 0.58 | 1.51, p = 0.29 | **no** | prediction, prior-dependent |
| `k96_0.1_decay24h` | 0.636 [0.594, 0.678], p = 0.01 | 0.01 | 12.6, p = 0.01 | **no** | prediction, prior-dependent: **rejected** |

**What the coupling adds**
1. **A1 is never explained by IUS evolution.** U is unselected on κ and β, so its genotypes keep the ancestral
   κ(13 h) and β. The null in U×U vials is therefore a property of the *ancestor* (hand-set κ = 0) or of the
   calibrated β, not something evolution produced. This holds whatever the priors (both runs).
2. **The null is consistent with a modest cost at 12–14 h.** With β from Byrne & Rice, costs of 0.1–0.3 still give
   courts-first, courts-most and latency statistics consistent with the thesis.
3. **A load-driven cost that is still large at 12–14 h is excluded.** The decay scenario has κ(13 h) ≈ 0.97 in U,
   and is rejected by courts-first, all-blocks and the latency infection term.
4. **New derived prediction.** If the cost tracks pathogen load, I females clear faster and carry a smaller cost
   at 13 h. Courts-first should then be less biased in I-female vials: U − I = +0.023 [−0.057, +0.108] in the decay
   scenario, against the observed −0.054 (p = 0.08). The sign is derived; the magnitude is prior-dependent.
5. **Derived null.** Preference is neutral in both regimes, so male genotype × infection should be absent. The
   model gives +0.001 to +0.020; observed −0.048, p = 0.14–0.29.
6. **Khan A2.** Male β varies by genotype around the Byrne & Rice mean: 0.479 [0.437, 0.525] against 0.463
   (p = 0.59). Genetic variance in β leaves both the mean (0.477 with β fixed) and the between-vial SD unchanged.
   **H8b stays rejected:** 0.110 vs 0.029.
7. **SA alleles.** Changing SA genotypes with κ, β and the cue fixed leaves courts-first, latency and courts-most
   bit-identical (regression test). They act on courtship only through the documented cue option, which is off by
   default.
8. **Unchanged.** Telegony stays non-genetic. H10, H4 and H14 keep their status.

## 9.6 Corrections made while building (disclosed)

1. **Trait variance prior.**
   - The prior was fixed before any run: the ancestral SD of each log trait is 0.15.
   - The first implementation set per-copy effects from the infinite-sites formula V = 4·N·U·σ². Finite-sites
     saturation (one-way mutation, mean derived frequency ≈ 0.2) gave a realised SD of 0.03–0.09 instead.
   - The burn-in is neutral for these traits, so the ancestor's genomes do not depend on the effect sizes. The
     effects are now rescaled to give exactly SD 0.15 in the expressing sex, without rerunning the burn-in.
   - This correction was prompted by an implementation check made after the first blocks' clearance responses
     had been seen. The raw variant is kept as the sensitivity run so that its effect is visible.
2. **Sign of the κ time course.** The first formula anchored the cost at 96 h and extrapolated backwards with
   exp(c·(96 − t)), which made faster-clearing genotypes costlier at 13 h. Corrected to load decaying from
   infection, κ ∝ exp(−c·t), with the ancestor normalised at 96 h. Test
   `test_kappa_and_hazard_fall_with_clearance_and_match_the_scenario_anchor` guards it.
3. **Reference means by expressing sex.** Preference (males) and tolerance (females) were first normalised over
   both sexes. The X counts once in males, so this biased β_rel to about 1.15. They are now normalised in the
   expressing sex.
4. **Two-sided F tails.** The latency-F rows first used an upper tail. That hid a model that predicts a large F
   when a small one is observed. They now use two-sided predictive tails.

Outputs from before these corrections were deleted, not kept.

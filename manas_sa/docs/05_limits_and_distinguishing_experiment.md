# 5. What the model cannot predict, and the smallest experiment that distinguishes SA from HRI

## 5.1 Not identified or not predictable (by design or by data)

| # | Quantity | Why the model cannot predict it | Consequence |
|---|---|---|---|
| L1 | The **absolute magnitude** of r_w,g,mf in LH | The mutation-class mixture (shares of SA, concordant and sex-limited mutations) is not reported anywhere. The model maps mixture → r, but the mixture is a free input. Only its *direction* is constrained, by the emulator inversion in `docs/04`. | We report which mixtures are *consistent* with r = 0.25–0.40, not a fitted mixture. |
| L2 | **Why** sex ratio changes r | Sex ratio enters only as a per-sex multiplier β_s(SR) on the breeding value, calibrated to line variances. There are no explicit harm, persistence or resistance traits. A true genotype × sex-ratio interaction (a change in the genetic correlation itself) is therefore **absent by construction**. | If the model reproduces the ordering, it shows that *attenuation* (lower genetic signal-to-noise at FB) is **sufficient**. It cannot show that a change in the genetic architecture of fitness across sex ratios is absent. The two are not identified with 39 lines (power in `docs/04`). |
| L3 | Am Nat 2025 Robertson covariances (target 3) | The paper is inaccessible. Trait data are not deposited. | We decompose the response of *fitness itself* (sex-specific FTNS: within-sex vs cross-sex covariance of relative fitness) from the raw data and the model. Trait-level covariances are **not identified**. |
| L4 | The sign of r in LHM-derived contrast studies, from first principles | Population history differs: LHM in Toronto/UCL/Uppsala vs LH in Mohali (>500 generations of separation, different maintenance), as do assay environments and fitness components (adult only vs lifetime). The model has none of these histories. | We can only say which assumption *would* move the sign (SA fraction, sampling, components; `docs/04`), not which one did. |
| L5 | Exact HRI magnitudes at N = 2,500 | Rescaled runs (λ = 2) inflate N·V_g by λ². λ = 4 collapses into neo-Y-like haplotype classes. | Sign and ordering are trusted at λ = 2. Magnitude is trusted only at λ = 1 (checked at 0.1 M). |
| L6 | Dominance-reversal balancing selection in the IBM | The IBM's SA alleles are additive and net-deleterious (b = 0.5), so SA polymorphism is maintained by mutation–selection–drift, not by balancing selection. | Balancing selection is handled only in the analytic modifier layer (target 5, Connallon–Clark recursions). |
| L7 | Gene conversion; recombination landscape within arms | Not modelled: uniform female crossover density and no conversion. | Short-range (<1 kb) LD is slightly overstated. The map bins used are much larger. |
| L8 | MCMCglmm heritabilities and random-effect variances (Table 2B) | Not re-fitted. The model has no day × line structure beyond vial and day noise. | Not scored (`results/coverage.csv`). |

## 5.2 The smallest experiment that distinguishes true SA from linkage/HRI

**Logic.** Both mechanisms give a negative cross-sex covariance for fitness, but they put it in different terms of

> COV_mf,W = Σ_i (p_i q_i/2) α_f,i α_m,i (direct: pleiotropy) + Σ_{i≠j} D_ij α_f,i α_m,j (indirect: LD).

Under purely sex-limited selection (HRI, or the deterministic sex-differential LD of docs/02 §2.2), the direct term is 0 and everything sits in D. Recombination without selection erodes D by a factor (1 − c_ij) per meiosis. Pleiotropy is untouched.

### 5.2.1 The obvious experiment fails: recombining hemigenomes in the lab (tested in silico)

*Idea.* Pass the hemigenome lines through k generations of female meiosis, then re-extract and re-assay them. The prediction would be that SA (the direct term) is unchanged while LD (the indirect term) decays.

*Result* (`scripts/run_distinguish.py`, docs/04 §4.6): in every HRI and sex-limited population the whole of r is LD (r_LE ≈ 0), yet 16 generations of female meiosis barely change r. The power to detect a change with 200 original + 200 recombinant lines is ≤ 0.18.

The reason is general. LD that the population maintains against recombination every generation must be between tightly linked loci (k·c ≪ 1 for any feasible k). **This experiment cannot distinguish SA from HRI.**

### 5.2.2 The smallest experiment that does: vary N, hold everything else

HRI-driven negative covariance scales with the product N_e·c: interference weakens as N_e grows. Pleiotropic SA covariance has no first-order dependence on N, and neither does the deterministic sex-differential LD (docs/02 §2.2).

**Design** (an extension of the LH maintenance protocol):
1. Split the base population into replicate lines maintained at small N (e.g. N = 250, ≥ 4 replicates) and at the standard N = 1,920, all at the same sex ratio and with the same culture regime.
2. After about 2N_small generations (≈ 500), extract 40 hemigenomes from each replicate.
3. Run the target-1 assay at the equal sex ratio.

**Prediction**

| Mechanism | Effect of small N on r |
|---|---|
| HRI | r becomes **more negative** in the small-N lines |
| SA (or deterministic sex-differential LD) | **no systematic shift** with N |
| Concordant mutation load | r shifts *upwards*: drift raises the load of shared deleterious alleles, which raises positive covariance |

So the *direction* of the shift is diagnostic.

**Model test of the HRI arm** (`results/ne_check.csv`). The preprint's per-generation parameters were used (0.1 M, μ = 7×10⁻⁷, s ~ Γ(0.3, 0.05), 25,000 generations, 6 replicates); only N changed.

| N | r_mf,W |
|---|---|
| 625 | −0.128 (sd 0.145) |
| 1,250 | −0.144 (sd 0.129) |
| 2,500 | −0.042 (sd 0.100) |

- The direction is as HRI predicts: smaller N gives more negative r, by about 0.09.
- The trend is not monotone.
- Replicate populations scatter so much that about **31 replicate populations per N** would be needed for 80% power, and that is before hemiclone-assay noise is added.

**Conclusion (C).** No feasible single experiment cleanly separates HRI from SA when |r| ≈ 0.1. The model does, however, give a practical **magnitude-and-sign criterion**:
- HRI and sex-limited LD produce only *negative* r:
  - population means |r| ≲ 0.15 at realistic map lengths (0.1–1 M), with or without male recombination;
  - single non-degenerate populations reach −0.26 at 0.1 M and −0.19 at 1 M;
  - −0.3 to −0.4 is reached only at ≤ 0.01 M.
- A positive r (LH, +0.25 to +0.40) cannot be HRI-driven, which makes HRI at most a small negative offset to a larger positive concordant term.
- Chippindale's −0.30 is within HRI's single-population range for short maps, so it is **not identified**.
- Values of −0.41 and −0.52 (Collet UU, Innocenti & Morrow) exceed what HRI produced at 0.1–1 M. A hemigenome averages over whole chromosome arms of about 0.5 M female map, so they point to SA (or collapse-like differentiation) rather than HRI (C).
- Values from −0.3 to 0 are ambiguous. For those, the N-replicate design above is the smallest experiment, and it is large.

**Cheaper partial versions** (weaker inference):
- **Genomic r_LE.** Sequence the hemigenomes and estimate per-locus female and male effects. Then r_LE = Σ(pq)α_fα_m/√(…). This is the direct-term-only correlation the model computes; in silico it separates the regimes perfectly. With 39 lines it is statistically infeasible, and it would need hundreds of lines with dense genotypes.

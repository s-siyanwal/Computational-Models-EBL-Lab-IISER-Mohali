# 10. Plain-language summary: hypothesis tests and the coupled model

A non-technical companion to `docs/08_hypothesis_tests.md` and `docs/09_coupling.md`. The numbers come from
`outputs/hypothesis_tests.csv` and `outputs/coupled_comparison.csv`.

---

## Part A: Does the model actually predict the results? (hypothesis tests)

### The question each test asks
For each published result: *if the world worked the way our model says, would we be surprised to see what the paper
saw?*
- **Not surprised (p above 0.05):** the result is *consistent* with the model.
- **Very surprised (p below 0.05):** the model is probably missing something.
- **Multiple tests.** About 40 tests were run, so a few will look surprising by luck. After correcting for that:
  - **rejected:** still surprising;
  - **tension:** surprising before the correction but not after.

A second question matters as much: **could this test have failed at all?** A test the model was bound to pass proves
nothing.

### 1. Real predictions that came out right (weak to moderate evidence)
- **Khan's mate-choice experiment.**
  - The model learned how choosy males are from a different study (Byrne & Rice) and predicted Khan's result blind.
  - It got the direction right and about two-thirds of the size.
  - However, "males don't choose at all" also fits fairly well. The data favour our model by only about 3 to 1.
- **Chinmay's courtship data** favour "infected females are no less fertile" over "40% less fertile", by about 3 to
  1.
- **Courts-most spread.** Calibrated on half of Chinmay's data, the model predicted the other half: 0.405 vs 0.423.
- **The linkage (Hill–Robertson) paper.**
  - At the paper's own population size: same direction, same ordering across map lengths, similar sizes.
  - The "collapsed" runs we had discarded turned out to be an artefact of scaling the simulation down, not biology.

### 2. Where the model fails (rejected)
- **Khan's vials vary far less than the model predicts** (0.029 vs 0.110). Either the CIs were computed differently
  than assumed, or the mating setup differs from the protocol.
- **The sex difference in offspring survival** in Chinmay's telegony experiment. Independent data say the sexes
  survive infection equally, and the model cannot produce the large difference he saw.

### 3. Warning signs (tension)
- Male and female genotype affect how fast males start courting. The model has no mechanism for this.
- In Manas's data, females' fitness rankings shift more between sex-ratio environments than the model allows.
- The choosiness value written into the original spec predicts too strong a preference for Khan's data.

### 4. "Consistent", but meaningless
- The null results ("males don't prefer healthy females", "the first mate doesn't affect offspring") were built in.
- Manas's correlations: the model's range runs from about −0.3 to +0.5, so almost anything passes.
- The LH value and the sex-ratio ordering were effectively put in, so they come back out.

### Bottom line
- **Genuine predictions** are right in direction and roughly in size. The evidence is modest because the experiments
  are small.
- **Where the data show something extra** (vial scatter, sex differences in survival, genotype effects), the model
  fails. That points to the biology it is missing.
- **Many earlier "matches" were not evidence,** because they could not fail.

---

## Part B: The coupled model

### What was built
Before, there were two separate models:
1. **The genetics model (Manas's).** It simulates fly populations evolving: genes, mutations, selection.
2. **The behaviour model (Chinmay's).** It simulates a male choosing between a healthy and an infected female.

The behaviour model used **fixed settings** typed in by hand, for example "infection costs 0% of fertility". Now
the genetics model **evolves** the flies, including:
- how well they fight infection;
- how much infection hurts their fertility;
- how choosy the males are.

Those evolved flies are then put into Chinmay's test. Information flows one way only, evolution → flies → choice
test, and neither model is tuned to make the other look right.

### How it works
1. **Ancestor:** a large ancestral population evolves for a long time.
2. **Split:** it splits into the two kinds of population used in the lab.
   - **I (infected):** every generation all flies are infected; only survivors breed (the real protocol).
   - **U (unhandled):** never infected.
3. **Selection:** 60 generations, 4 replicate pairs.
4. **Choice test:** flies from the evolved populations go through Chinmay's experiment.

**What-if versions.** The fertility cost of infection at 12–14 hours, when Chinmay tested, was never measured. So
four versions were run instead of guessing:
- no cost;
- 10%;
- 30%;
- a cost that is huge early and fades as the female clears the infection.

### Findings
1. **Evolution cannot explain Chinmay's result.**
   - U flies were never infected, so they were never selected for anything related to infection. They keep what
     the ancestor had.
   - Whatever made males ignore infection must already have been true in the ancestor. This holds in every version.
2. **His null result is consistent with a small cost.** Male choosiness, calibrated independently, is fairly weak.
   Even a 10–30% fertility loss would usually still show no preference. So "no preference" does not prove "no
   cost".
3. **A large early cost is ruled out.** If infected females were nearly infertile at 12–14 h, males would clearly
   prefer healthy ones (64% vs the observed 52%). The data reject that.
4. **Two new testable predictions.**
   - If the cost tracks bacterial load, I females (who clear infection faster) should be *less* avoided than U
     females. A larger experiment is needed to check this.
   - I and U males should not differ in choosiness. Chinmay's data agree.
5. **Khan's experiment is still predicted reasonably** (0.479 vs 0.463).
6. **Earlier failures are still failures:** vial scatter, the telegony sex difference, genotype effects on
   courtship speed, and sex-ratio effects in females.
7. **The rules hold (tested):**
   - a sexually antagonistic gene cannot change courtship unless it changes fertility cost or male perception;
   - a stepfather's genes do not affect offspring;
   - males never recombine in any *Drosophila* run (no reruns were needed).

### The honest caveat
I flies evolve better survival, the right direction. **How much** better depends on how much genetic variation the
ancestor had, and that is unmeasured:
- with one assumption, about 25% better (real value about 37%);
- with another, only 1–2% better.

That number is illustrative, not a prediction. None of the courtship conclusions depend on it.

### Bottom line
- The coupled model does not explain Chinmay's result through evolution. It shows why it cannot.
- **Ruled in:** a small fertility cost. **Ruled out:** a large early one.
- **Experiment that would settle it:** measure the fertility of infected females 12–14 hours after infection.

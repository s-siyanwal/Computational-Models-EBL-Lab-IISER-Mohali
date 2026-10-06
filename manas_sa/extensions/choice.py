"""Module A, part 2: male courtship choice.

Kernel (spec 4.2):
    P(court female 1) = exp(beta*q1) / (exp(beta*q1) + exp(beta*q2)),
    beta = beta0 * depletion        (1 after 12 h sperm depletion, lower if not)
If q1 = q2 the probability is 0.5 for every beta.

Two-choice vial (Chinmay 2019 Ch. 3; Wittman & Fedorka 2015): minute-by-minute
courtship over the observation period.
  * onset ~ Exp(mean = latency_mean * exp(-slope * (max(q1,q2) - 1)))
    times optional male-genotype / female-genotype intercept multipliers
    (no infection interaction);
  * the first target is drawn from the kernel -> courts first (CF);
  * each later minute the male courts with prob courting_prob; the target is the
    previous one with prob `persistence`, otherwise redrawn from the kernel;
  * courtship latency (CL) to each female = time of the first minute aimed at
    her (onset time for the first target); never-courted females are censored;
  * courts most (CM) = share of courting minutes spent on the sham female.
Copulation duration is a separate draw that ignores infection unless the
postcopulatory flag is on (off for Chinmay and Khan).

Group mating assay (Khan & Prasad 2013): 10 males, 10 infected + 10 sham intact
females, 45 min, single mating per fly. Each free male picks among the still
unmated females with the same kernel; a courted female accepts with her
receptivity. Mating-bias score = infected mated / all mated.
"""
from __future__ import annotations

import numpy as np

from . import require
from . import stats as S
from .quality import Female, perceived_quality, receptivity

# thesis Table 3.2 (p.19, PDF p.37): block, treatment "AxB" = B male given
# infected vs sham A females, K = sham courted first, N = vials
TABLE_3_2 = [
    (1, "UXU", 18, 39), (1, "IXI", 21, 32), (1, "IXU", 23, 38), (1, "UXI", 12, 26),
    (2, "UXU", 13, 32), (2, "IXI", 17, 34), (2, "UXI", 15, 27), (2, "IXU", 15, 31),
    (3, "UXU", 14, 29), (3, "IXI", 15, 29), (3, "UXI", 14, 31), (3, "IXU", 6, 17),
    (4, "UXU", 9, 18), (4, "IXI", 18, 33), (4, "UXI", 18, 28), (4, "IXU", 12, 19),
]


def beta(P, depleted):
    return P["beta0"] * (P["depletion_depleted"] if depleted else P["depletion_nondepleted"])


def p_first(q1, q2, b):
    """Softmax probability of courting female 1."""
    return 1.0 / (1.0 + np.exp(-b * (np.asarray(q1, float) - np.asarray(q2, float))))


def copulation_duration(rng, n, infected, P):
    """Arbitrary unit (mean 1, sd 0.15); only differences are scored."""
    shift = 0.0
    if P.get("postcopulatory_flag", False):
        shift = -P.get("postcopulatory_effect", 0.0) * float(infected)
    return rng.normal(1.0 + shift, 0.15, n)


def simulate_two_choice(rng, n, q_sham, q_inf, b, P, lat_mult=1.0):
    """Vectorised over n vials. q_sham, q_inf and b are scalars, or arrays of
    length n (one female pair and one male per vial, from a genotype record
    via extensions/coupler.py). Returns dict of arrays: cf_sham (bool),
    lat_sham, lat_inf (minutes; nan = never courted), cm_sham (nan if no
    courtship)."""
    T = int(P["observation_min"] / P["interval_min"])
    dt = P["interval_min"]
    p_sham = np.asarray(p_first(q_sham, q_inf, b), float)
    mean_lat = P["latency_mean_min"] * np.exp(-P["latency_quality_slope"] * (np.maximum(q_sham, q_inf) - 1.0)) * lat_mult
    onset = rng.exponential(mean_lat, n)
    target = np.where(rng.random(n) < p_sham, 0, 1)          # 0 = sham, 1 = infected
    cf_sham = target == 0
    lat = np.full((n, 2), np.nan)
    lat[np.arange(n), target] = onset
    court = np.zeros((n, 2))
    active = onset < P["observation_min"]
    start = np.minimum(np.ceil(onset / dt).astype(int), T)
    court[np.arange(n), target] += active
    cur = target.copy()
    for t in range(1, T):
        on = active & (t >= start)
        courting = on & (rng.random(n) < P["courting_prob"])
        redraw = courting & (rng.random(n) >= P["persistence"])
        new_t = np.where(rng.random(n) < p_sham, 0, 1)
        cur = np.where(redraw, new_t, cur)
        idx = np.where(courting)[0]
        court[idx, cur[idx]] += 1
        first = courting & np.isnan(lat[np.arange(n), cur])
        lat[np.where(first)[0], cur[first]] = (t + 0.5) * dt
    total = court.sum(1)
    cm_sham = np.where(total > 0, court[:, 0] / np.maximum(total, 1), np.nan)
    cf = np.where(active, cf_sham, False)
    return dict(cf_sham=cf, courted=active, lat_sham=np.where(active, lat[:, 0], np.nan),
                lat_inf=np.where(active, lat[:, 1], np.nan), cm_sham=cm_sham)


def simulate_chinmay_study(rng, P, flags, kappa=None, cue_on=None):
    """One replicate of thesis Ch. 3 (4 blocks x 4 treatments, Table 3.2 sizes).
    kappa / cue_on override the regime (A3 / A4). Returns per-vial records."""
    require(flags, "choice")
    require(flags, "quality")
    Pq = dict(P)
    if kappa is not None:
        Pq["kappa"] = kappa
    if cue_on is not None:
        Pq["cue_on"] = cue_on
    b = beta(Pq, Pq["male_depleted"])
    parts = {k: [] for k in ("block", "trt", "female", "male", "courted", "cf_sham", "lat_sham", "lat_inf", "cm_sham")}
    for block, trt, _, n in TABLE_3_2:
        fem_gt, male_gt = trt[0], trt[-1]
        q_s = perceived_quality(Female(fem_gt, False, Pq["decapitated"]), Pq)
        q_i = perceived_quality(Female(fem_gt, True, Pq["decapitated"]), Pq)
        mult = np.exp(Pq["male_genotype_latency_effect"] * (1 if male_gt == "I" else -1)
                      + Pq["female_genotype_latency_effect"] * (1 if fem_gt == "I" else -1))
        r = simulate_two_choice(rng, n, q_s, q_i, b, Pq, lat_mult=mult)
        parts["block"].append(np.full(n, block))
        parts["trt"].append(np.full(n, trt))
        parts["female"].append(np.full(n, fem_gt))
        parts["male"].append(np.full(n, male_gt))
        for k in ("courted", "cf_sham", "lat_sham", "lat_inf", "cm_sham"):
            parts[k].append(r[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def analyse_chinmay(recs):
    """Apply the thesis's tests to (real or simulated) per-vial records."""
    import pandas as pd
    d = pd.DataFrame(recs)
    d = d[d.courted]
    out = {}
    for blk, g in d.groupby("block"):                       # spec A1: per block, two-sided
        out[("cf_p_block", blk)] = S.binom_two_sided(g.cf_sham.sum(), len(g))
    for (blk, trt), g in d.groupby(["block", "trt"]):        # thesis: per block x treatment
        out[("cf_p_two_sided", blk, trt)] = S.binom_two_sided(g.cf_sham.sum(), len(g))
        out[("cf_p_min_tail", blk, trt)] = S.binom_min_tail(g.cf_sham.sum(), len(g))
        out[("cm_mean", blk, trt)] = float(g.cm_sham.mean())
        out[("cm_sd", blk, trt)] = float(g.cm_sham.std(ddof=1))
        from scipy import stats
        out[("cm_t_p", blk, trt)] = float(stats.ttest_1samp(g.cm_sham.dropna(), 0.5).pvalue)
    out["cf_sham_share"] = float(d.cf_sham.mean())
    out["cm_mean_all"] = float(d.cm_sham.mean())
    for blk, g in d.groupby("block"):
        out[("cm_mean_block", blk)] = float(g.cm_sham.mean())
    # courtship latency ANOVA: two rows per vial (latency to the sham and to the
    # infected female), factors infection status, female and male genotype, block
    long = pd.concat([d.assign(lat=d.lat_sham, inf="sham"), d.assign(lat=d.lat_inf, inf="infected")])
    long = long[np.isfinite(long.lat)]
    a = S.factorial_anova(long.lat.values, {"INF": long.inf.values, "FEMALES": long.female.values,
                                            "MALE": long.male.values}, block=long.block.values)
    for term, (F, p, *_) in a.items():
        out[("cl", term)] = (F, p)
    cm = d[np.isfinite(d.cm_sham)]
    a2 = S.factorial_anova(cm.cm_sham.values, {"FEMALES": cm.female.values, "MALES": cm.male.values},
                           block=cm.block.values)
    for term, (F, p, *_) in a2.items():
        out[("cm_anova", term)] = (F, p)
    return out


def group_mating_vial(rng, q_a, q_b, b, n_males, n_each, minutes, acceptance, latency_mean):
    """Group mating assay with two female types A and B (n_each of each) and
    n_males males, single mating per fly. Each minute every free male starts a
    courtship with prob 1 - exp(-1/latency_mean); his target is drawn among the
    still-unmated females with the softmax kernel; the female accepts with
    probability `acceptance`. Returns (mated_A, mated_B)."""
    avail = np.array([n_each, n_each])
    mated = np.zeros(2, int)
    free_males = n_males
    p_try = 1 - np.exp(-1.0 / latency_mean)
    for _ in range(int(minutes)):
        if free_males == 0 or avail.sum() == 0:
            break
        for _ in range(rng.binomial(free_males, p_try)):
            if avail.sum() == 0:
                break
            w = avail * np.exp(b * np.array([q_a, q_b]))
            k = 0 if rng.random() < w[0] / w.sum() else 1
            if rng.random() < acceptance:
                avail[k] -= 1
                mated[k] += 1
                free_males -= 1
    return int(mated[0]), int(mated[1])


def khan_group_vial(rng, P, b):
    """One Khan & Prasad (2013) mating-bias vial. Returns (n_inf_mated,
    n_sham_mated, copulation durations of infected, of sham)."""
    q_s = perceived_quality(Female("LH", False, P["decapitated"]), P)
    q_i = perceived_quality(Female("LH", True, P["decapitated"]), P)
    acc = receptivity(Female("LH", False, P["decapitated"]), P)
    ms, mi = group_mating_vial(rng, q_s, q_i, b, P["group_males"], P["group_females_each"],
                               P["group_minutes"], acc, P["latency_mean_min"])
    cd_i = copulation_duration(rng, mi, True, P)
    cd_s = copulation_duration(rng, ms, False, P)
    return mi, ms, cd_i, cd_s


def simulate_khan(rng, P, flags, n_experiments=4):
    """Khan & Prasad (2013) design: 4 replicate experiments x 6 vials.
    Returns per-experiment mean bias scores, pooled CD difference (inf - sham)
    and the two-choice courts-first probability under the same parameters."""
    require(flags, "choice")
    require(flags, "quality")
    b = beta(P, P["male_depleted"])
    scores, cdi, cds = [], [], []
    for _ in range(n_experiments):
        s = []
        for _ in range(int(P["vials_per_experiment"])):
            mi, ms, ci, cs = khan_group_vial(rng, P, b)
            if mi + ms:
                s.append(mi / (mi + ms))
            cdi.extend(ci)
            cds.extend(cs)
        scores.append(float(np.mean(s)))
    q_s = perceived_quality(Female("LH", False, False), P)
    q_i = perceived_quality(Female("LH", True, False), P)
    return dict(bias_scores=scores, cd_diff=float(np.mean(cdi) - np.mean(cds)),
                cd_se=float(np.sqrt(np.var(cdi, ddof=1) / len(cdi) + np.var(cds, ddof=1) / len(cds))),
                p_sham_first=float(p_first(q_s, q_i, b)))

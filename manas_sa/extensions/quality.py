"""Module A, part 1: female quality.

State of a female: genotype G (I, U, S, ...), infection I in {0, 1},
decapitated flag. From the regime parameters:

  survival hazard   h = dose * (1 - c(G)) per hour (if infected), with clearance
                    c(G) expressed relative to the U population (c_U = 0) and
                    derived from the reported 96 h mortalities (no OD curve)
  fecundity         F = F0 * (1 - kappa * I)
  perceived quality q = F / F0            (cue on)
                    q = 1                 (cue off: no perceivable difference)
                    q *= S(96 h)          (only if couple_quality_to_survival)
  receptivity       0 if decapitated, else receptivity_intact
Decapitation removes receptivity, never q.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOURS = 96.0


@dataclass(frozen=True)
class Female:
    genotype: str
    infected: bool
    decapitated: bool = False


def _hazard_from_mortality(m, hours=HOURS):
    return -np.log(max(1e-12, 1.0 - m)) / hours


def mortality_96h(genotype, infected, P):
    if not infected:
        return P["mortality_96h_sham"]
    m_u = P["mortality_96h_U"]
    if genotype == "I":
        return max(0.0, m_u - P["survival_advantage_I"])
    return m_u                          # U, S, BRB, LH: no evolved advantage


def clearance(genotype, P):
    """c(G) relative to U (c_U = 0): the fraction of U's hazard that genotype G
    removes, so that h_G = dose * h_U * (1 - c(G))."""
    h_u = _hazard_from_mortality(mortality_96h("U", True, P))
    h_g = _hazard_from_mortality(mortality_96h(genotype, True, P))
    return 1.0 - h_g / h_u


def hazard(female, P):
    if not female.infected:
        return _hazard_from_mortality(P["mortality_96h_sham"])
    h_u = _hazard_from_mortality(P["mortality_96h_U"])
    return P["dose"] * h_u * (1.0 - clearance(female.genotype, P))


def survival(female, P, hours=HOURS):
    return float(np.exp(-hazard(female, P) * hours))


def fecundity(female, P):
    return P["F0"] * (1.0 - P["kappa"] * float(female.infected))


def perceived_quality(female, P):
    if not P.get("cue_on", True):
        return 1.0
    q = fecundity(female, P) / P["F0"]
    if P.get("couple_quality_to_survival", False):
        q *= survival(female, P)
    return q


def receptivity(female, P):
    return 0.0 if female.decapitated else P.get("receptivity_intact", 1.0)

"""Single source of truth for every experiment.

Both engines (deterministic oracle and IBM) receive their *conditions* from
here: viability tables, recombination rate, inheritance mode, population
size, initial state, seeds. Nothing else is shared.

PROFILE controls the computational budget: 'quick' for smoke runs (~minutes),
'full' for the report (~tens of minutes on 8 cores).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

PROFILE = os.environ.get("SC_PROFILE", "full")
MASTER_SEED = 20261005
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _q(quick, full):
    return quick if PROFILE == "quick" else full


def seed_for(name: str) -> int:
    """Deterministic, name-derived seed so each experiment is reproducible
    independently of execution order."""
    return int(np.random.SeedSequence([MASTER_SEED, *map(ord, name)]).generate_state(1)[0] % 2**31)


# --------------------------------------------------------------------------
# Viability tables (the 'conditions' handed to the IBM)
# --------------------------------------------------------------------------
def poster_viability(a, b, k1, k2):
    """wtab[sex, haplotype]; sex 0 = female, 1 = male; haplotypes
    [A1M1, A1M2, A2M1, A2M2]. Straight from the poster's fitness table."""
    return np.array([[1 + b * k1, 1.0, 1 + b * k2, 1 + b],
                     [1 + a, 1 + a, 1.0, 1.0]])


def owen_viability(a1, h1, b1, a2, h2, b2):
    """wtab[sex, 3*mito + nA] for the diploid kernel; mito irrelevant.
    nA = number of A alleles: 0 -> aa (b), 1 -> Aa (h), 2 -> AA (a)."""
    f = [b1, h1, a1]
    m = [b2, h2, a2]
    return np.array([f + f, m + m], float)


def xlinked_viability(a1, h1, b1, a2, b2):
    """wtab[sex, class]; females by number of A (aa, Aa, AA); males by A on X."""
    return np.array([[b1, h1, a1], [b2, a2, 0.0]], float)


def cytonuclear_viability(W):
    """Flatten W[sex, mito, genotype] -> wtab[sex, 3*mito+g]."""
    return np.asarray(W, float).reshape(2, 6)


def validate_viability(wtab):
    wtab = np.asarray(wtab, float)
    if not np.all(np.isfinite(wtab)) or np.any(wtab < 0):
        raise ValueError(f"viabilities must be finite and non-negative: {wtab}")
    if np.any(wtab.max(1) <= 0):
        raise ValueError("each sex needs at least one viable class")
    return wtab


# --------------------------------------------------------------------------
# Experiment definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PosterCfg:
    b_strong: float = 0.2
    b_weak: float = 0.02
    r_bif: float = 0.1
    k_pairs: tuple = ((0.2, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.8),
                      (0.8, 0.2), (0.5, 1.0), (0.0, 0.5))
    k_grid3: tuple = (0.2, 0.5, 0.8)
    heat_n: int = _q(21, 51)              # (k1,k2) resolution of heatmaps
    ar_n: int = _q(41, 101)               # (a,r) resolution for each cell
    away_ic: tuple = (0.5, 0.75, 0.0)     # (pA1, pM1, D) -- see report
    near_eps: float = 1e-3                # M1 perturbation 'near' fixed point
    # IBM
    one_locus_N: tuple = (500, 5000)
    one_locus_reps: int = _q(8, 24)
    N_inv: int = _q(2000, 5000)
    reps_inv: int = _q(16, 40)
    inv_m0: float = 0.02
    inv_horizon: int = 300                # generations over which early spread is compared
    persist_N: tuple = _q((500, 2000), (500, 2000, 8000))
    persist_a: int = _q(7, 15)
    persist_reps: int = _q(16, 48)
    persist_gens: int = _q(1000, 2000)
    inv_kpairs: tuple = ((0.2, 0.2), (0.5, 0.5), (0.8, 0.2))
    inv_a: int = _q(4, 7)
    inv_r: tuple = _q((0.02, 0.2, 0.5), (0.02, 0.1, 0.2, 0.35, 0.5))
    N_bif: int = _q(1000, 1500)
    bif_a: int = _q(5, 13)
    bif_reps: int = _q(8, 24)
    bif_gens: int = _q(2000, 5000)
    near_m0_ibm: float = 0.02
    thr_k: tuple = (0.2, 0.2)
    thr_a: tuple = (0.19, 0.205)
    thr_m0: tuple = _q((0.05, 0.15, 0.3, 0.5), (0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65))
    thr_N: tuple = _q((500, 2000), (500, 2000, 6000))
    thr_reps: int = _q(16, 48)
    thr_gens: int = _q(2000, 4000)
    basin_det_n: int = _q(300, 600)
    basin_kpairs: tuple = ((0.2, 0.2), (0.0, 0.5))
    basin_ibm_n: int = _q(24, 96)
    basin_a: int = _q(4, 7)
    basin_N: int = _q(1000, 1500)
    basin_gens: int = _q(2000, 4000)
    conv_k: tuple = (0.5, 0.5)
    conv_a: float = 0.22
    conv_ic: tuple = (0.5, 0.3, 0.0)
    conv_gens: int = 300
    conv_N: tuple = _q((250, 1000, 4000), (250, 500, 1000, 2000, 4000, 8000, 16000))
    conv_reps: int = _q(16, 64)
    anim_k: tuple = (0.2, 0.2)
    anim_a: float = 0.2
    anim_N: int = 2000
    anim_gens: int = _q(300, 900)


@dataclass(frozen=True)
class OwenCfg:
    N: int = _q(1000, 2000)
    reps: int = _q(48, 160)
    init_copies: int = 20
    gens: int = _q(200, 400)
    grid_n: int = _q(5, 9)
    ratio_lo: float = 0.9
    ratio_hi: float = 1.1
    # Parsons' worked example: beta1 = 0.05, beta2 = -0.02, h = 1
    example: tuple = (0.98, 1.0, 0.95, 0.98, 1.0, 1.02)   # a1,h1,b1,a2,h2,b2
    # an Owen (1953)-type bistable parameter set found by random search
    bistable: tuple = (0.725, 1.239, 0.854, 1.228, 0.623, 1.053)
    phase_N: int = _q(1000, 4000)
    phase_ics: int = _q(5, 9)           # grid of initial (p1,p2) per axis
    phase_reps: int = _q(8, 24)
    phase_gens: int = _q(150, 300)


@dataclass(frozen=True)
class CurseCfg:
    N: int = _q(1000, 4000)
    reps: int = _q(32, 96)
    x0: float = 0.05
    gens: int = _q(300, 600)
    s_f: float = 0.02
    s_m_values: tuple = (0.0, 0.3, 0.6, 0.9)  # s_m = 1 -> extinction once fixed
    twofold_sf: tuple = _q((0.05, 0.5, 1.0, 1.5), (0.02, 0.1, 0.3, 0.6, 0.9, 1.0, 1.1, 1.4, 2.0))
    twofold_N: int = _q(1000, 2000)
    twofold_x0: float = 0.02
    twofold_gens: int = _q(300, 500)
    weak_N: int = _q(100, 200)
    weak_x0: float = 0.1
    weak_gens: int = _q(1500, 3000)
    weak_reps: int = _q(200, 800)
    restorer: dict = field(default_factory=lambda: dict(s_f=0.02, s_m=0.5, restore=1.0,
                                                        h_R=0.5, cost_R=0.01))
    restorer_x0: float = 0.05
    restorer_p0: float = 0.02
    restorer_gens: int = _q(800, 1500)


POSTER = PosterCfg()
OWEN = OwenCfg()
CURSE = CurseCfg()


# --------------------------------------------------------------------------
# Extended suite: classical SA results + drift theory + ML analyses
# --------------------------------------------------------------------------
def kidwell_autosomal_viability(s_f, s_m, h_f, h_m, counted="A1"):
    """wtab[sex, nA] for the diploid kernel, Kidwell/Rice/Fry fitness table.
    counted = which allele the kernel counts ('A1' male-beneficial or 'A2')."""
    f = [1.0, 1 - h_f * s_f, 1 - s_f]     # by number of A1: 0, 1, 2
    m = [1 - s_m, 1 - h_m * s_m, 1.0]
    if counted == "A2":
        f, m = f[::-1], m[::-1]
    return np.array([f + f, m + m], float)


def kidwell_xlinked_viability(s_f, s_m, h_f, counted="A1"):
    """wtab[sex, class] for the X-linked kernel (females by number of counted
    alleles, males by counted allele on their X)."""
    f = [1.0, 1 - h_f * s_f, 1 - s_f]
    m = [1 - s_m, 1.0]
    if counted == "A2":
        f, m = f[::-1], m[::-1]
    return np.array([f, m + [0.0]], float)


def xy_viability(s_f, s_m, h_f, h_m):
    """wtab[sex, #A1] for the XY kernel (sex 0 = XX, 1 = XY)."""
    return np.array([[1.0, 1 - h_f * s_f, 1 - s_f], [1 - s_m, 1 - h_m * s_m, 1.0]], float)


def genic_viability(s_f, s_m):
    """Additive (genic) sex-specific selection on allele 'A': 1, 1+s, 1+2s."""
    return np.array([[1.0, 1 + s_f, 1 + 2 * s_f] * 2, [1.0, 1 + s_m, 1 + 2 * s_m] * 2], float)


@dataclass(frozen=True)
class ExtCfg:
    # E1/E2 protected-polymorphism maps
    scenarios: tuple = (("additive", 0.5, 0.5), ("dominance reversal h=0.25", 0.25, 0.25),
                        ("male-benefit recessive (h_f=0.2, h_m=0.8)", 0.2, 0.8))
    s_lo: float = 0.02
    s_hi: float = 0.3
    grid_auto: int = _q(7, 12)
    grid_x: int = _q(5, 8)
    N_map: int = 2000
    reps_map: int = _q(16, 48)
    reps_map_x: int = _q(12, 32)
    dose: float = 0.01
    T_map: int = 300
    # E3 Kimura
    kim_N: tuple = _q((200, 500), (200, 500, 800))
    kim_S: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0)        # S = 4 N s_bar
    kim_p0: float = 0.05
    kim_reps: int = _q(200, 800)
    kim_sa_spread: float = 0.05                                 # s_f = s+d, s_m = s-d
    # M2 emulator training set
    emu_n: int = _q(40, 160)
    emu_reps: int = _q(100, 300)
    # E4 XY linkage
    xy_r: tuple = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5)
    xy_sf: float = 0.1
    xy_sm: tuple = _q((0.04, 0.1, 0.2), (0.03, 0.06, 0.09, 0.12, 0.15, 0.2, 0.25, 0.3))
    xy_N: int = 2000
    xy_reps: int = _q(8, 16)
    xy_T: int = _q(500, 1000)
    # E5 diffusion persistence
    dif_a: tuple = (0.18, 0.2, 0.22, 0.24)
    dif_N: tuple = _q((50, 100, 200), (50, 100, 200, 400))
    dif_reps: int = _q(100, 400)
    # M3 inverse problem
    inv_N: tuple = (500, 5000)
    inv_train: int = _q(150, 600)
    inv_test: int = _q(40, 120)
    inv_T: int = 100
    inv_every: int = 5


EXT = ExtCfg()

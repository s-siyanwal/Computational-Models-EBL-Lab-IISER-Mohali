"""One command for the whole pipeline.

    python scripts/run_all.py                    # default profile, re-uses cached stages
    python scripts/run_all.py --profile quick    # smoke run (~30 min on a 4-core laptop)
    python scripts/run_all.py --only table       # just rebuild results/comparison_table.csv

Stages (each skipped when its output exists, unless --force):
  recompute  analysis/recompute_bmc2022.py   raw data -> published statistics (+10k bootstrap)
  modifier   scripts/run_modifier.py         authors' recursion code, exact grid
  lh         scripts/run_lh.py               LH regimes + calibrated hemiclone assays
  design     scripts/run_design.py           architecture sweep + power (needs lh)
  emulator   scripts/run_emulator.py         GP emulator, sensitivity, inversion (needs design)
  hri        scripts/run_hri.py              sex-limited selection + linkage grid
  distinguish scripts/run_distinguish.py     SA-vs-HRI recombination experiment + power (needs lh, hri)
  table      scripts/make_comparison.py      comparison table, coverage, figures
  chinmay    scripts/run_chinmay.py          optional extensions: Chinmay 2019 / Khan 2013 / Morrow 2003 scoring
All heavy stages run at below-normal priority with (cpu - 2) numba threads.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def stages(profile):
    py = sys.executable
    return [
        ("recompute", [py, "analysis/recompute_bmc2022.py", "10000"], "bmc2022_recomputed.csv"),
        ("modifier", [py, "scripts/run_modifier.py"], "modifier_claims.csv"),
        ("lh", [py, "scripts/run_lh.py", "--profile", profile], f"lh_assays_{profile}.csv"),
        ("design", [py, "scripts/run_design.py", "--profile", profile] +
         (["--points", "4", "--reps", "2", "--n_assay", "6", "--n_power", "6"] if profile == "quick" else
          ["--points", "12", "--reps", "2"]), f"power_sexratio_{profile}.csv"),
        ("emulator", [py, "scripts/run_emulator.py", "--profile", profile], f"emulator_inversion_{profile}.csv"),
        ("hri", [py, "scripts/run_hri.py", "--profile", profile], f"hri_runs_{profile}.csv"),
        ("distinguish", [py, "scripts/run_distinguish.py", "--profile", profile], f"distinguish_{profile}.csv"),
        ("table", [py, "scripts/make_comparison.py"], None),
        ("byrne", [py, "scripts/calibrate_byrne_rice.py"], "../outputs/byrne_rice_calibration.csv"),
        ("calibrated", [py, "scripts/run_chinmay_calibrated.py"], "../outputs/chinmay_calibrated.csv"),
        ("chinmay", [py, "scripts/run_chinmay.py"] + (["--reps", "40"] if profile == "quick" else []),
         "../outputs/chinmay_comparison.csv"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", choices=["quick", "default", "full"])
    ap.add_argument("--only", default="", help="comma-separated stage names")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    os.makedirs(RES, exist_ok=True)
    for name, cmd, out in stages(args.profile):
        if only and name not in only:
            continue
        if out and not args.force and os.path.exists(os.path.join(RES, out)) and not only:
            print(f"[{name}] cached ({out})")
            continue
        print(f"[{name}] {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        with open(os.path.join(RES, f"stage_{name}_{args.profile}.log"), "w") as log:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{name}] exit {rc} in {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()

"""Master orchestration: smoke checks -> experiments -> summary JSON -> report.

usage:  python scripts/run_all.py [--profile quick|full] [--only poster,owen,curse]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["quick", "full"])
    ap.add_argument("--only", default="poster,owen,curse,ext")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "opencl"],
                    help="IBM backend: numba CPU kernels or OpenCL GPU kernels (X-linked stays on CPU)")
    ap.add_argument("--steps", default="", help="comma list of step-id prefixes, e.g. P4,P5,O1 (default: all)")
    args = ap.parse_args()
    os.environ["SC_PROFILE"] = args.profile
    os.environ["SC_BACKEND"] = args.backend
    # share the machine politely: leave CPU threads free for other work and
    # run at below-normal priority (override with SC_THREADS)
    os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("SC_THREADS", str(max(1, (os.cpu_count() or 2) - 2))))
    try:
        import psutil  # optional
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass

    from sexconflict import io
    from sexconflict.viz import style

    style.apply()
    log = io.setup_logging()
    env = io.environment()
    from sexconflict.ibm import runners
    env["ibm_backend"] = runners.backend().__name__.rsplit(".", 1)[-1]
    if env["ibm_backend"] == "opencl_backend":
        env["gpu_device"] = runners.backend().device_name()
    log.info("environment: %s", env)
    groups = {}
    if "poster" in args.only:
        from sexconflict.experiments import poster_exp
        groups["poster"] = poster_exp.ALL
    if "owen" in args.only:
        from sexconflict.experiments import owen_exp
        groups["owen"] = owen_exp.ALL
    if "curse" in args.only:
        from sexconflict.experiments import curse_exp
        groups["curse"] = curse_exp.ALL

    if "ext" in args.only:
        from sexconflict.experiments import extended_exp
        groups["extended"] = extended_exp.ALL

    summary_path = os.path.join(io.C.RESULTS_DIR, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        import json
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    summary["environment"] = env
    for g, exps in groups.items():
        summary.setdefault(g, {})
        summary[g]["environment"] = dict(env)  # backend/profile used for this suite
        steps = [x for x in args.steps.split(",") if x]
        for name, fn in exps:
            if steps and not any(name.startswith(x) for x in steps):
                continue
            t0 = time.time()
            log.info("[%s] %s ...", g, name)
            try:
                summary[g][name] = fn()
            except Exception as exc:  # keep going; record the failure
                import traceback
                log.error("[%s] %s FAILED: %s %s", g, name, exc, traceback.format_exc())
                summary[g][name] = {"error": repr(exc)}
            summary[g][name + " (seconds)"] = round(time.time() - t0, 1)
            log.info("[%s] %s done in %.1fs", g, name, time.time() - t0)
            io.save_json("summary", summary)
    log.info("all done")


if __name__ == "__main__":
    main()

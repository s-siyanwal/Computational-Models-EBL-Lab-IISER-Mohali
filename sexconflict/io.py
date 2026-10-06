"""Storage: tables -> Parquet (pandas/pyarrow), arrays -> compressed npz,
run metadata -> JSON. Everything lands under results/."""
from __future__ import annotations

import json
import logging
import os
import platform
import time

import numpy as np
import pandas as pd

from . import config as C

DATA = os.path.join(C.RESULTS_DIR, "data")
FIGS = os.path.join(C.RESULTS_DIR, "figures")
ANIM = os.path.join(C.RESULTS_DIR, "animations")
for d in (DATA, FIGS, ANIM):
    os.makedirs(d, exist_ok=True)

log = logging.getLogger("sexconflict")


def setup_logging():
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(os.path.join(C.RESULTS_DIR, "run.log"), "a", "utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def save_table(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(DATA, f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def load_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA, f"{name}.parquet"))


def save_arrays(name: str, **arrays) -> str:
    path = os.path.join(DATA, f"{name}.npz")
    np.savez_compressed(path, **arrays)
    return path


def fig_path(name):
    return os.path.join(FIGS, f"{name}.png")


def anim_path(name):
    return os.path.join(ANIM, f"{name}.gif")


def environment():
    import matplotlib, numba, scipy, sympy
    return dict(python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                sympy=sympy.__version__, matplotlib=matplotlib.__version__, pandas=pd.__version__,
                profile=C.PROFILE, master_seed=C.MASTER_SEED, cpu_count=os.cpu_count(),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))


def _clean(o):
    """Recursively make keys strings (tuples -> 'a, b') and values JSON-able."""
    if isinstance(o, dict):
        return {(", ".join(map(str, k)) if isinstance(k, tuple) else str(k)): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def save_json(name, obj):
    """Atomic write (temp file + rename) so a crash never leaves broken JSON."""
    path = os.path.join(C.RESULTS_DIR, f"{name}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=2, default=_jsonable)
    os.replace(tmp, path)
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)

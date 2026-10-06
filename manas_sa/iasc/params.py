"""Load config/default.toml (single source of truth) and set polite threading."""
from __future__ import annotations

import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 2)))


def _intkeys(d):
    if isinstance(d, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _intkeys(v) for k, v in d.items()}
    return d


def load(path=None):
    with open(path or os.path.join(ROOT, "config", "default.toml"), "rb") as f:
        return _intkeys(tomllib.load(f))


def below_normal_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass

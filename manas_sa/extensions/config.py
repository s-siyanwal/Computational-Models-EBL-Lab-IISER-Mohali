"""Regime configuration (config/ius.yaml, config/khan.yaml).

Every parameter is a mapping with ``value`` and ``provenance`` (and usually
``grade``). A loader without the pyyaml dependency is included; it accepts the
block-style subset used by these files: nested mappings by 2-space indentation,
scalars (int, float, true/false, null, quoted or bare strings) and inline lists
``[a, b]``. If pyyaml is installed it is used instead.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")


def _scalar(tok):
    t = tok.strip()
    if t == "":
        return None
    if (t[0] == t[-1]) and t[0] in "\"'" and len(t) >= 2:
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        return [] if not inner else [_scalar(x) for x in inner.split(",")]
    low = t.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


def _strip_comment(line):
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def parse_yaml_subset(text):
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"odd indentation: {raw!r}")
        key, sep, rest = line.strip().partition(":")
        if not sep:
            raise ValueError(f"expected 'key: value': {raw!r}")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip() == "":
            child = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(rest)
    return root


def load_regime(name):
    """Load config/<name>.yaml. Returns dict with 'modules' (flags) and
    'parameters' ({name: {'value', 'provenance', ...}})."""
    path = name if os.path.isfile(name) else os.path.join(CONFIG_DIR, f"{name}.yaml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml  # optional
        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = parse_yaml_subset(text)
    params = cfg.get("parameters", {})
    for k, v in params.items():
        if not isinstance(v, dict) or "value" not in v or "provenance" not in v:
            raise ValueError(f"parameter {k!r} in {path} needs 'value' and 'provenance'")
    cfg.setdefault("modules", {})
    return cfg


def values(cfg):
    """Flat {name: value} view of the parameters."""
    return {k: v["value"] for k, v in cfg["parameters"].items()}

"""Make the manas_sa packages (iasc, extensions, analysis, scripts) importable
wherever pytest is started from.

Without this, `python -m pytest tests manas_sa/tests` run from the repository
root (the quick start in the top-level README) fails at collection with
`ModuleNotFoundError: No module named 'iasc'`: only the current directory is
put on sys.path, and the manas_sa packages live one level down.
"""
import pathlib
import sys

MANAS_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(MANAS_ROOT) not in sys.path:
    sys.path.insert(0, str(MANAS_ROOT))
